from lib import Model, clean

from transformers import AutoTokenizer # pyright: ignore[reportMissingTypeStubs]
from torch import load, no_grad, Tensor, softmax, argmax # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available

device = "cuda" if is_available() else "cpu"
print(device)

max_len = 128
amount = 3

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("beomi/kcELECTRA-base") # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
print("model loading")
model = Model(amount).to(device)
_ = model.load_state_dict(load("model.pth", weights_only=True, map_location=device)) # pyright: ignore[reportAny]
_ = model.eval()
print("model loaded")
chats: list[str] = []
for i in range(3):
  _ = chats.append(clean(input(f"chat {i+1}: ")))

_ = model.eval()
with no_grad():
  tokenizer_result: list[dict[str, Tensor]] = [
    tokenizer( # pyright: ignore[reportCallIssue]
      chat, add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt'
    ) for chat in chats
  ]
  input_ids: list[Tensor] = [t['input_ids'] for t in tokenizer_result]
  attention_mask: list[Tensor] = [t['attention_mask'] for t in tokenizer_result]
  outputs: Tensor = model(input_ids, attention_mask)
  probabilities = softmax(outputs, dim=0)
  prediction = argmax(probabilities, dim=0)
  risk_score = probabilities[1].item()
  print(f"Prediction: {'suicidal' if prediction == 1 else 'non-suicidal'}")
  print(f"Risk score: {risk_score}")

