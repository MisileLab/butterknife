from collections import defaultdict
from pathlib import Path
from pickle import dumps

from libraries.scrape import read, Data
from libraries.data import clean

from torch import Tensor
from torch.cuda import is_available
from transformers import ElectraTokenizer # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained('beomi/kcELECTRA-base') # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
data = read("data.avro")
datas: tuple[list[dict[str, list[Tensor]]], list[int]] = ([], [])
max_len = 128
amount = 3
device = "cuda" if is_available() else "cpu"

# just for data collection
real_len: list[int] = []
blank_tokenizer = 3
t = tqdm(data.to_dicts())
skipped = 0

for _i in t:
  i = Data.model_validate(_i)
  t_data: dict[str, list[Tensor]] = defaultdict(list)
  count = 0
  for j in i.data:
    data = clean(j)
    if data.replace('\n', '').strip() == '':
      continue
    encoded: dict[str, Tensor] = tokenizer( # pyright: ignore[reportArgumentType, reportAssignmentType, reportUnknownMemberType]
      data, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt'
    ).to(device)
    input_ids: Tensor = encoded['input_ids'].squeeze()
    attention_mask: Tensor = encoded['attention_mask'].squeeze()
    t_data['input_ids'].append(input_ids)
    t_data['attention_mask'].append(attention_mask)
    count += 1
    if max(len(input_ids[input_ids != blank_tokenizer]), len(input_ids[input_ids != blank_tokenizer])) == max_len:
      skipped += 1
      continue
    avg_len = (len(input_ids[input_ids != blank_tokenizer]) + len(input_ids[input_ids != blank_tokenizer])) // 2
    real_len.append(avg_len)
    t.set_description_str(f"{sum(real_len) / len(real_len):.3f}, {real_len.count(max_len)}, {skipped}")
  if count < amount:
    continue
  datas[0].append(t_data)
  datas[1].append(1 if i.suicidal else 0)

print(f"average length: {sum(real_len) / len(real_len)}")
print(f"skipped: {skipped}")
_ = Path("tokenized.pkl").write_bytes(dumps(datas))
