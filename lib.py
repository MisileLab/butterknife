from collections import defaultdict
from contextlib import suppress
from os import getenv
from pathlib import Path
from re import compile
from sys import stdout

try:
  from typing import Any, final, override
except ImportError:
  from typing_extensions import Any, final, override

from loguru import logger
from pandas import DataFrame, Series, concat # pyright: ignore[reportMissingTypeStubs]
from pandas import read_pickle as _read_pickle # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel
from torch import Tensor, nn, cat
from torch.cuda import is_available
from transformers import AutoModel # pyright: ignore[reportMissingTypeStubs]
from twscrape import API # pyright: ignore[reportMissingTypeStubs]
from emoji import replace_emoji
from soynlp.normalizer import repeat_normalize # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

logger.remove()
_ = logger.add(stdout, level="DEBUG")

pattern = compile(r'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = compile(
  r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)
device = "cuda" if is_available() else "cpu"

@final
class Model(nn.Module):
  def __init__(self, data_amount: int, pretrained_model: str = "beomi/kcelectra-base") -> None:
    super().__init__()  # pyright: ignore[reportUnknownMemberType]
    self.electra: AutoModel = AutoModel.from_pretrained(pretrained_model).to(device) # pyright: ignore[reportUnknownMemberType]
    hidden_size: int = int(self.electra.config.hidden_size * data_amount) # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    self.emotion_layer = nn.Sequential(
      nn.Linear(hidden_size, 128),
      nn.ReLU(),
      nn.Dropout(0.2)
    )

    self.context_layer = nn.Sequential(
      nn.Linear(hidden_size, 128),
      nn.ReLU(),
      nn.Dropout(0.2)
    )

    self.classifier = nn.Sequential(
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(64, 2)
    )

  @override
  def forward(self, input_ids: list[Tensor], attention_mask: list[Tensor]) -> Tensor:
    outputs: dict[str, list[Any]] = defaultdict(list) # pyright: ignore[reportExplicitAny]
    for sub_input_ids, sub_attention_mask in zip(input_ids, attention_mask):
      sub_input_ids = sub_input_ids.to(device)
      sub_attention_mask = sub_attention_mask.to(device)
      output: Any = self.electra( # pyright: ignore[reportCallIssue, reportExplicitAny, reportAny]
        input_ids=sub_input_ids,
        attention_mask=sub_attention_mask,
        return_dict=True
      )

      sequence_output = output.last_hidden_state # pyright: ignore[reportAny]
      pooled_output = sequence_output[:, 0, :] # pyright: ignore[reportAny]

      outputs["sequence_output"].append(sequence_output)
      outputs["pooled_output"].append(pooled_output)

    emotion_features: Tensor = self.emotion_layer(
      cat([cat(t, dim=0) for t in outputs["pooled_output"]]) # pyright: ignore[reportAny]
    )
    context_features: Tensor = self.context_layer(
      cat([cat(t).mean(dim=1) for t in outputs["sequence_output"]]) # pyright: ignore[reportAny]
    )

    combined_features = cat([emotion_features, context_features], dim=1).to(device)
    logits: Tensor = self.classifier(combined_features)
    return logits

class User(BaseModel):
  uid: int
  name: str
  suicidal: bool
  url: str

class Data(User):
  data: list[str]
  confirmed: bool = False

def clean(x: str) -> str: 
  x = pattern.sub(' ', x)
  x = replace_emoji(x, replace='') #emoji 삭제
  x = url_pattern.sub('', x)
  x = x.strip()
  x = repeat_normalize(x, num_repeats=2)
  return x

def get_proxy():
  proxy_url = getenv("PROXY_URL")
  proxy_user = getenv("PROXY_USERNAME")
  proxy_pass = getenv("PROXY_PASSWORD")

  prx = (
    None if None in [proxy_url, proxy_user, proxy_pass] else
    f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
  )
  logger.debug(prx)
  return prx

api = API(proxy=get_proxy())
# api.pool._order_by = "RANDOM()"

async def get_usernames() -> list[str]:
  lst = await api.pool.accounts_info()
  return [a["username"] for a in lst if not (a["active"] or a["logged_in"])]

def read_pickle(file_path: str) -> DataFrame:
  if Path(file_path).exists():
    _df = _read_pickle(file_path)
    df = DataFrame() if not isinstance(_df, DataFrame) else _df
  else:
    df = DataFrame()
  return df

def write_to_pickle(df: DataFrame, file_path: str) -> None:
  with suppress(ValueError):
    df = df.reset_index()
  with suppress(KeyError):
    del df["level_0"]
  with suppress(KeyError):
    del df["index"]
  df.to_pickle(file_path)

def is_unique(df: DataFrame, key: str, value: object) -> bool:
  try:
    return df.loc[df[key] == value].empty # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
  except KeyError:
    return True

def append(df: DataFrame, data: dict[str, object] | Series | BaseModel) -> DataFrame:
  if isinstance(data, BaseModel):
    data = Series(data.model_dump())
  elif isinstance(data, dict):
    data = Series(data)
  return concat([df, data.to_frame().T], ignore_index=True)
