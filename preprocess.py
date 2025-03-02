from collections import defaultdict
from pathlib import Path
from pickle import dumps

from libraries.scrape import read, Data
from libraries.data import clean

from torch import Tensor
from transformers import AutoTokenizer # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('beomi/kcELECTRA-base') # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
data = read("data.avro")
datas: tuple[list[dict[str, list[Tensor]]], list[int]] = ([], [])
max_len = 128
amount = 3

for _i in tqdm(data.to_dicts()):
  i = Data.model_validate(_i)
  t_data: dict[str, list[Tensor]] = defaultdict(list)
  count = 0
  for t in i.data:
    data = clean(t)
    if data.replace('\n', '').strip() == '':
      continue
    encoded: dict[str, Tensor] = tokenizer( # pyright: ignore[reportUnknownVariableType]
      data, add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt'
    )
    t_data['input_ids'].append(encoded['input_ids'].flatten()) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    t_data['attention_mask'].append(encoded['attention_mask'].flatten()) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    count += 1
  if count < amount:
    continue
  datas[0].append(t_data)
  datas[1].append(1 if i.suicidal else 0)

_ = Path("tokenized.pkl").write_bytes(dumps(datas))

