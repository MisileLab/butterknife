from collections import defaultdict
from pathlib import Path
from pickle import dumps

from lib import read_pickle, Data, clean

from torch import Tensor
from transformers import AutoTokenizer # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm
from os import environ

environ["HF_HOME"] = "./.cache"

tokenizer = AutoTokenizer.from_pretrained('beomi/kcELECTRA-base') # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
data = read_pickle("data.pkl")
datas: list[tuple[dict[str, list[Tensor]], bool]] = []
max_len = 128

for _i in tqdm(data.to_dict('records')): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]
  i = Data.model_validate(_i)
  t_data: dict[str, list[Tensor]] = defaultdict(list)
  for t in i.data:
    data = clean(t)
    encoded: dict[str, Tensor] = tokenizer( # pyright: ignore[reportUnknownVariableType]
      data, add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt'
    )
    t_data['input_ids'].append(encoded['input_ids'].flatten()) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    t_data['attention_mask'].append(encoded['attention_mask'].flatten()) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
  datas.append((t_data, i.suicidal))

_ = Path("tokenized.pkl").write_bytes(dumps(datas))

