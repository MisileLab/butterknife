from pathlib import Path
from pickle import loads

from tqdm.auto import tqdm

from torch import tensor, Tensor, nn, cat, optim, save, abs, zeros, ones, no_grad # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available, empty_cache
from torch.utils.data import DataLoader, TensorDataset, random_split

from lib import Model

device = "cuda" if is_available() else "cpu"
data_device = "cpu" # dataset is too big to fit in GPU memory
print(device)

amount = 3
batch_size = 32

print("loading dataset")
dataset: SuicideDataset = loads(Path("dataset.pkl"))
print("loaded")
lx_t = len(dataset)
train_size = int(0.6 * lx_t)
val_size = int(0.2 * lx_t)
test_size = lx_t - val_size - train_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
del dataset, val_size, train_size, test_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

epoches = 300
model = Model(amount).to(device)
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

t = tqdm(range(epoches))

# Early Stopping Parameters
patience = 5
best_loss = float('inf')
patience_counter = 0
model_path = "model.pth"
best_model = model.state_dict()

for epoch in t:
  _ = model.train()
  running_loss = 0.0
  count = 0
  for _x, _y in train_loader:
    _x: Tensor
    _y: Tensor
    x = _x.to(device)
    y = _y.to(device)
    if x.size(0) < amount:
      continue
    optimizer.zero_grad()

    windows_x = x.unfold(0, amount, 1).transpose(1, 2)
    windows_y = y[amount - 1:]

    del x, y, _x, _y

    output: Tensor = model(windows_x)
    target = windows_y.mean(dim=1, keepdim=True)

    loss: Tensor = criterion(output, target)
    _ = loss.backward() # pyright: ignore[reportUnknownMemberType]
    _ = optimizer.step() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    running_loss += loss.item() * windows_x.size(0)
    count += windows_x.size(0)
    del windows_x, windows_y, output, target, loss
  
  avg_loss = running_loss / count
  t.set_description(str(avg_loss))
  
  if avg_loss < best_loss:
    best_loss = avg_loss
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
  test_x_list: list[Tensor] = []
  test_y_list: list[Tensor] = []
  for x_batch, y_batch in tqdm(test_loader, desc="Collecting test data"):
    x_batch: Tensor
    y_batch: Tensor
    _ = test_x_list.append(x_batch)
    _ = test_y_list.append(y_batch)
  test_x_all = cat(test_x_list, dim=0).to(device)
  test_y_all = cat(test_y_list, dim=0).to(device)
  
  if test_x_all.size(0) < amount:
    print("Not enough test samples to form a window.")
  else:
    windows_x = test_x_all.unfold(0, amount, 1).transpose(1, 2)
    windows_y = test_y_all[amount - 1:]
    
    predictions: Tensor = model(windows_x)
    targets = windows_y.mean(dim=1, keepdim=True)
    error = abs(predictions - targets)
    avg_error = error.mean().item()
    print("Average test error:", avg_error)
