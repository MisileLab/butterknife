from sys import version_info
from os import environ
from pathlib import Path

if version_info.major >= 3 and version_info.minor >= 12:
  from typing import final, override
else:
  from typing_extensions import final, override

from torch import nn, Tensor, cat
from transformers import ElectraModel # pyright: ignore[reportMissingTypeStubs]

environ["HF_HOME"] = str(Path("./.cache").absolute())

@final
class Model(nn.Module):
  def __init__(self, data_amount: int, pretrained_model: str = "beomi/kcELECTRA-base") -> None:
    super().__init__()  # pyright: ignore[reportUnknownMemberType]
    self.electra: ElectraModel = ElectraModel.from_pretrained(pretrained_model) # pyright: ignore[reportUnknownMemberType]
    hidden_size: int = self.electra.config.hidden_size * data_amount # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    features_size = 64 * data_amount

    self.emotion_layer = nn.Sequential(
      nn.Linear(hidden_size, features_size), # pyright: ignore[reportUnknownArgumentType]
      nn.ReLU(),
      nn.Dropout(0.2)
    )

    self.context_layer = nn.Sequential(
      nn.Linear(hidden_size, features_size), # pyright: ignore[reportUnknownArgumentType]
      nn.ReLU(),
      nn.Dropout(0.2)
    )

    self.percent_layer = nn.Sequential(
      nn.Linear(hidden_size * 2, features_size), # pyright: ignore[reportUnknownArgumentType]
      nn.ReLU(),
      nn.Linear(features_size, 1),
      nn.Sigmoid()
    )

  @override
  def forward(self, input_ids: list[Tensor], attention_mask: list[Tensor]) -> Tensor:
    sequence_outputs: list[Tensor] = []
    pooled_outputs: list[Tensor] = []

    for sub_input_ids, sub_attention_mask in zip(input_ids, attention_mask):
      sub_input_ids = sub_input_ids
      sub_attention_mask = sub_attention_mask
      output: dict[str, Tensor] = self.electra(
        input_ids=sub_input_ids,
        attention_mask=sub_attention_mask,
        return_dict=True
      )

      sequence_outputs.append(output['last_hidden_state'].mean(dim=1))
      pooled_outputs.append(output['last_hidden_state'][:, 0, :])

    combined_features = cat([*sequence_outputs, *pooled_outputs], dim=1)
    return self.percent_layer(combined_features).squeeze() # pyright: ignore[reportAny]

