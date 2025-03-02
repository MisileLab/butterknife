#!pip install scikit-learn

from pathlib import Path
from pickle import loads
from sys import version_info
if version_info.major >= 3 and version_info.minor > 11:
  from typing import final, override
else:
  from typing_extensions import final, override

from tqdm.auto import tqdm
import torch
from torch import Tensor, nn, optim, save, no_grad, tensor # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available, empty_cache
from torch.utils.data import DataLoader, Dataset as tDataset
from sklearn.model_selection import train_test_split # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

from libraries.model import Model

device = "cuda" if is_available() else "cpu"
print(device)

amount = 3

@final
class Dataset(tDataset): # pyright: ignore[reportMissingTypeArgument]
  def __init__(self, data: list[dict[str, list[Tensor]]], labels: list[int]):
    _data: list[dict[str, list[Tensor]]] = []
    _labels: list[Tensor] = []
    for i, j in enumerate(tqdm(data)):
      total = len(j['input_ids'])-amount-1
      if total <= 0:
        continue
      j['input_ids'] = [k.to(device) for k in j['input_ids']]
      j['attention_mask'] = [k.to(device) for k in j['attention_mask']]
      current = 0
      while current <= total:
        _data.append({
          'input_ids': j['input_ids'][current:current+3],
          'attention_mask': j['attention_mask'][current:current+3]
        })
        _labels.append(tensor(labels[i], dtype=torch.float))
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
test_size = 0.1

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
  batch_size=1,
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
    labels = tensor([batch['label'][0], batch['label'][0]]).to(device)

    optimizer.zero_grad()
    outputs: Tensor = model(input_ids, attention_mask).to(device) # pyright: ignore[reportAny, reportRedeclaration]
    loss: Tensor = criterion(outputs, labels)
    _ = loss.backward() # pyright: ignore[reportUnknownMemberType]
    _ = optimizer.step() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    train_loss += loss.item()
    _, predicted = torch.max(outputs, dim=0)
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
  total_percent_error: list[Tensor] = []
  total_label_miss = 0
  for batch in tqdm(test_loader, desc="Collecting test data"): # pyright: ignore[reportUnknownArgumentType]
    input_ids = [i.to(device) for i in batch['input_ids']]
    attention_mask = [i.to(device) for i in batch['attention_mask']]
    label = batch['label'][0].to(device)
    outputs: Tensor = model(input_ids, attention_mask)
    probabilities = torch.softmax(outputs, dim=0)
    prediction = torch.argmax(probabilities, dim=0)
    risk_score = probabilities[1].item()
    if prediction != label:
      total_label_miss += 1
    total_percent_error.append(torch.abs(label - probabilities[1]))
  avg_error = tensor(total_percent_error).mean()
  print("average percent error:", tensor(total_percent_error).mean())
  print("total label miss:", total_label_miss)
  print("total data:", len(test_loader)) # pyright: ignore[reportUnknownArgumentType]
  print("accuracy:", 1 - total_label_miss/len(test_loader)) # pyright: ignore[reportUnknownArgumentType]
