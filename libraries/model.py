from sys import version_info
from collections import defaultdict
from os import environ
from pathlib import Path

if version_info.major >= 3 and version_info.minor > 10:
  from typing import final, override, Any
else:
  from typing_extensions import final, override, Any

from torch import nn, Tensor, cat
from torch.cuda import is_available
from transformers import AutoModel # pyright: ignore[reportMissingTypeStubs]

device = "cpu" if is_available() else "cuda"
environ["HF_HOME"] = str(Path("./.cache").absolute())

@final
class Model(nn.Module):
  def __init__(self, data_amount: int, pretrained_model: str = "beomi/kcELECTRA-base", device: str = device) -> None:
    super().__init__()  # pyright: ignore[reportUnknownMemberType]
    self.electra: AutoModel = AutoModel.from_pretrained(pretrained_model).to(device) # pyright: ignore[reportUnknownMemberType]
    hidden_size: int = int(self.electra.config.hidden_size) # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    features_size = 128
    self.emotion_layer = nn.Sequential( # pyright: ignore[reportUnannotatedClassAttribute]
      nn.Linear(hidden_size, features_size),
      nn.ReLU(),
      nn.Dropout(0.2)
    )

    self.context_layer = nn.Sequential( # pyright: ignore[reportUnannotatedClassAttribute]
      nn.Linear(hidden_size, features_size),
      nn.ReLU(),
      nn.Dropout(0.2)
    )

    self.classifier = nn.Sequential( # pyright: ignore[reportUnannotatedClassAttribute]
      nn.Linear(features_size * 2 * data_amount, 64 * data_amount), # idk why 32
      nn.Linear(64 * data_amount, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(64, 2)
    )

  @override
  def forward(self, input_ids: list[Tensor], attention_mask: list[Tensor]) -> Tensor:
    outputs: dict[str, list[Tensor]] = defaultdict(list)
    for sub_input_ids, sub_attention_mask in zip(input_ids, attention_mask):
      sub_input_ids = sub_input_ids.to(device)
      sub_attention_mask = sub_attention_mask.to(device)
      output: Any = self.electra( # pyright: ignore[reportCallIssue, reportExplicitAny, reportAny]
        input_ids=sub_input_ids,
        attention_mask=sub_attention_mask,
        return_dict=True
      )

      sequence_output: Tensor = output.last_hidden_state # pyright: ignore[reportAny]
      pooled_output = sequence_output[:, 0, :]

      outputs["sequence_output"].append(sequence_output)
      outputs["pooled_output"].append(pooled_output)

    emotion_features: Tensor = self.emotion_layer(
      cat(outputs["pooled_output"])
    )
    context_features: Tensor = self.context_layer(
      cat([t.mean(dim=1) for t in outputs["sequence_output"]])
    )

    combined_features = cat([emotion_features, context_features], dim=1).to(device).flatten()
    logits: Tensor = self.classifier(combined_features)
    return logits

