from contextlib import suppress
from os import getenv
from pathlib import Path
from re import compile
from sys import stdout
from typing import final, override

from loguru import logger
from pandas import DataFrame, Series, concat # pyright: ignore[reportMissingTypeStubs]
from pandas import read_pickle as _read_pickle # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel
from torch import Tensor, cat, nn
from twscrape import API # pyright: ignore[reportMissingTypeStubs]
from emoji import replace_emoji
from soynlp.normalizer import repeat_normalize # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

logger.remove()
_ = logger.add(stdout, level="DEBUG")

pattern = compile(r'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

@final
class Model(nn.Module):
  def __init__(self, data_amount: int) -> None:
    super().__init__()  # pyright: ignore[reportUnknownMemberType]
    self.embedding_layers = nn.ModuleList([
      nn.Sequential(
        nn.Linear(3072, 64),
        nn.ReLU(),
        nn.Dropout(0.2)
      ) for _ in range(data_amount)
    ])

    self.percentage_layer = nn.Sequential()
    n = 64 * data_amount
    while n // 4 >= 32:
      _ = self.percentage_layer.append(nn.Linear(n, n // 4))
      _ = self.percentage_layer.append(nn.ReLU())
      _ = self.percentage_layer.append(nn.Dropout(0.2))
      n //= 4
    _ = self.percentage_layer.append(nn.Linear(n, 1))
    self.percentage_layer = self.percentage_layer

  @override
  def forward(self, x: Tensor) -> Tensor:
    embedding_layer_results: list[Tensor] = []
    for i, f in enumerate(self.embedding_layers):
      xi = x[:, i, :]
      _ = embedding_layer_results.append(f(xi)) # pyright: ignore[reportAny]
    concatenated = cat(embedding_layer_results, dim=1)
    return self.percentage_layer(concatenated) # pyright: ignore[reportAny]

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
