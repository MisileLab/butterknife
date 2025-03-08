from enum import Enum
from os import getenv, environ
from pathlib import Path
from sys import stdout

from loguru import logger
from polars import DataFrame, Series, col, concat, from_dicts, read_avro
from pydantic import BaseModel
from twscrape import API # pyright: ignore[reportMissingTypeStubs]

environ["HF_HOME"] = str(Path("./.cache").absolute())

class UserType(Enum):
  normal = 0
  suicidal = 1
  ignored = 2

class User(BaseModel):
  uid: int
  name: str
  user_type: UserType
  url: str

class Data(User):
  data: list[str]
  confirmed: bool = False

logger.remove()
_ = logger.add(stdout, level="DEBUG")

def get_proxy():
  proxy_url = getenv("PROXY_URL")
  proxy_user = getenv("PROXY_USERNAME")
  proxy_pass = getenv("PROXY_PASSWORD")

  prx = (
    None if None in [proxy_url, proxy_user, proxy_pass] else
    f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
  )
  logger.debug(prx)
  return prx

api = API(proxy=get_proxy())
# api.pool._order_by = "RANDOM()"

async def get_usernames() -> list[str]:
  lst = await api.pool.accounts_info()
  return [a["username"] for a in lst if not (a["active"] or a["logged_in"])]

async def get_inactives() -> list[str]:
  return [a["username"] for a in await api.pool.accounts_info() if not a["active"]]

def read(file_path: str) -> DataFrame:
  return read_avro(file_path) if Path(file_path).exists() else DataFrame()

def is_unique(df: DataFrame, key: str, value: object) -> bool:
  return df.filter(col(key) == value).is_empty()

def append(df: DataFrame, data: dict[str, object] | Series | BaseModel | DataFrame) -> DataFrame:
  if isinstance(data, BaseModel):
    data = from_dicts([data.model_dump()])
  elif isinstance(data, dict):
    data = from_dicts([data])
  return concat([df, DataFrame(data)], how="vertical", rechunk=True)
