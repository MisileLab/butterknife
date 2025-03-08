from collections import defaultdict
from pathlib import Path
from pickle import dumps

from libraries.scrape import UserType, read, Data
from libraries.data import clean

from torch import Tensor
from transformers import AutoTokenizer # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('beomi/kcELECTRA-base') # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
data = read("data.avro")
datas: tuple[list[dict[str, list[Tensor]]], list[int]] = ([], [])
max_len = 128
amount = 3

# just for data collection
real_len: list[int] = []
blank_tokenizer = 3
t = tqdm(data[data["user_type"] != UserType.ignored].to_dicts())
skipped = 0

for _i in t:
  i = Data.model_validate(_i)
  t_data: dict[str, list[Tensor]] = defaultdict(list)
  count = 0
  for j in i.data:
    data = clean(j)
    if data.replace('\n', '').strip() == '':
      continue
    encoded: dict[str, Tensor] = tokenizer( # pyright: ignore[reportUnknownVariableType]
      data, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids: Tensor = encoded['input_ids'].squeeze() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    attention_mask: Tensor = encoded['attention_mask'].squeeze() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    t_data['input_ids'].append(input_ids) # pyright: ignore[reportUnknownArgumentType]
    t_data['attention_mask'].append(attention_mask) # pyright: ignore[reportUnknownArgumentType]
    count += 1
    if max(len(input_ids[input_ids != blank_tokenizer]), len(input_ids[input_ids != blank_tokenizer])) == max_len: # pyright: ignore[reportUnknownArgumentType]
      skipped += 1
      continue
    avg_len = (len(input_ids[input_ids != blank_tokenizer]) + len(input_ids[input_ids != blank_tokenizer])) // 2 # pyright: ignore[reportUnknownArgumentType]
    real_len.append(avg_len)
    t.set_description_str(f"{sum(real_len) / len(real_len):.3f}, {real_len.count(max_len)}, {skipped}")
  if count < amount:
    continue
  datas[0].append(t_data)
  datas[1].append(i.user_type.value)

print(f"average length: {sum(real_len) / len(real_len)}")
print(f"skipped: {skipped}")
_ = Path("tokenized.pkl").write_bytes(dumps(datas))
