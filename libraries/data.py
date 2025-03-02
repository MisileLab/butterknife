from os import environ
from pathlib import Path
from re import compile

from emoji import replace_emoji
from soynlp.normalizer import repeat_normalize # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

environ["HF_HOME"] = str(Path("./.cache").absolute())

pattern = compile(r'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = compile(
  r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)

def clean(x: str) -> str: 
  x = pattern.sub(' ', x)
  x = replace_emoji(x, replace='') #emoji 삭제
  x = url_pattern.sub('', x)
  x = x.strip()
  x = repeat_normalize(x, num_repeats=2)
  return x

