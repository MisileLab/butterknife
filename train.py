#!pip install scikit-learn loguru pandas pydantic transformers twscrape emoji soynlp

from pathlib import Path
from pickle import loads
try:
  from typing import final, override
except ImportError:
  from typing_extensions import final, override

from tqdm.auto import tqdm
import torch
from torch import Tensor, nn, optim, save, no_grad, tensor # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available, empty_cache
from torch.utils.data import DataLoader, Dataset as tDataset
from sklearn.model_selection import train_test_split # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

from lib import Model

device = "cuda" if is_available() else "cpu"
print(device)

amount = 3
batch_size = 32

@final
class Dataset(tDataset): # pyright: ignore[reportMissingTypeArgument]
  def __init__(self, data: list[dict[str, list[Tensor]]], labels: list[int]):
    _data: list[dict[str, list[Tensor]]] = []
    _labels: list[Tensor] = []
    for i, j in enumerate(data):
      total = len(j['input_ids'])-amount-1
      if total <= 0:
        continue
      j['input_ids'] = [k.to(device) for k in j['input_ids']]
      j['attention_mask'] = [k.to(device) for k in j['attention_mask']]
      t = tqdm(desc=f'{i}/{len(data)}', total=total)
      current = 0
      while current <= total:
        _data.append({
          'input_ids': j['input_ids'][current:current+3],
          'attention_mask': j['attention_mask'][current:current+3]
        })
        _labels.append(tensor(labels[i]))
        _ = t.update()
        current += 1
      j['input_ids'] = [k.to('cpu') for k in j['input_ids']]
      j['attention_mask'] = [k.to('cpu') for k in j['attention_mask']]
    self.data = _data
    self.labels = _labels
    del data, labels
    assert len(self.data) == len(self.labels)

  def __len__(self):
    return len(self.data)

  @override
  def __getitem__(self, index: int) -> dict[str, list[Tensor]]:
    data = self.data[index]
    data["label"] = [self.labels[index]]
    return data

print("loading dataset")
raw_dataset: tuple[list[dict[str, list[Tensor]]], list[int]] = loads(Path("tokenized.pkl").read_bytes())
print("loaded")

random_state = 1379357662
test_size = 0.2

train_data, test_data, train_labels, test_labels = train_test_split( # pyright: ignore[reportUnknownVariableType]
  raw_dataset[0], raw_dataset[1],
  test_size=test_size,
  random_state=random_state,
  stratify=raw_dataset[1]
)
del random_state, test_size

train_dataset = Dataset(train_data, train_labels) # pyright: ignore[reportUnknownArgumentType]
test_dataset = Dataset(test_data, test_labels) # pyright: ignore[reportUnknownArgumentType]
train_loader = DataLoader( # pyright: ignore[reportUnknownVariableType]
  train_dataset,
  batch_size=batch_size,
  shuffle=True
)
test_loader = DataLoader( # pyright: ignore[reportUnknownVariableType]
  test_dataset,
  batch_size=1,
  shuffle=False
)
del train_dataset, test_dataset

epoches = 300
model = Model(amount).to(device)
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Early Stopping Parameters
patience = 3
best_loss = float('inf')
patience_counter = 0
model_path = "model.pth"
best_model = model.state_dict()

t = tqdm(range(epoches))
for epoch in t:
  _ = model.train()

  train_loss = 0
  train_correct = 0
  train_total = 0
  
  for batch in train_loader:
    batch: dict[str, list[Tensor]]
    input_ids = [i.to(device) for i in batch['input_ids']]
    attention_mask = [i.to(device) for i in batch['attention_mask']]
    labels = batch['label'][0].to(device)
    
    optimizer.zero_grad()
    outputs: Tensor = model(input_ids, attention_mask)
    loss: Tensor = criterion(outputs, labels)
    _ = loss.backward() # pyright: ignore[reportUnknownMemberType]
    _ = optimizer.step() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    
    train_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    train_total += labels.size(0)
    train_correct += (predicted == labels).sum().item()

  train_accuracy = 100 * train_correct / train_total
  t.set_description(f'{train_loss/len(train_loader)}, {train_accuracy}') # pyright: ignore[reportUnknownArgumentType]

  if train_loss < best_loss:
    best_loss = train_loss/len(train_loader) # pyright: ignore[reportUnknownArgumentType]
    patience_counter = 0
    best_model = model.state_dict()
  else:
    patience_counter += 1
    if patience_counter >= patience:
      print("Early stopping triggered!")
      break

print("Training completed!")

_ = save(best_model, model_path)
_ = model.load_state_dict(best_model)

empty_cache()

_ = model.eval()
with no_grad():
  total_error: list[Tensor] = []
  for x_batch, y_batch in tqdm(test_loader, desc="Collecting test data"): # pyright: ignore[reportUnknownArgumentType]
    x_batch: dict[str, Tensor]
    y_batch: Tensor
    predictions: Tensor = model(x_batch['input_ids'].to('cuda'), x_batch['attention_mask'].to('cuda'))
    total_error.append(torch.abs(predictions - y_batch))
  avg_error = tensor(total_error).mean()
  print("average test error:", avg_error[0], avg_error[1])
  print("just for debug:", len(avg_error))
