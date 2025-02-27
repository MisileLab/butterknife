from lib import Model

from torch import load # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available

device = "cuda" if is_available() else "cpu"
print(device)

amount = 3

model = Model(amount).to(device)
_ = model.load_state_dict(load("model.pth", weights_only=True, map_location=device)) # pyright: ignore[reportAny]
_ = model.eval()
chats: list[str] = []
for i in range(3):
  _ = chats.append(input(f"chat {i+1}: "))

# print(run_model(model, chats))

