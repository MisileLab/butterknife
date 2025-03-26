from os import environ
from pathlib import Path
from sys import stdout

from loguru import logger
from polars import DataFrame, Series, col, concat, from_dicts, read_avro
from pydantic import BaseModel

environ["HF_HOME"] = str(Path("./.cache").absolute())

logger.remove()
_ = logger.add(stdout, level="DEBUG")

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
