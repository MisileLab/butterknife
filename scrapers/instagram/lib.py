from time import sleep

from instagrapi import Client # pyright: ignore[reportMissingTypeStubs]
from instagrapi.exceptions import ( # pyright: ignore[reportMissingTypeStubs]
  ClientConnectionError,
  ClientForbiddenError,
  ClientLoginRequired,
  ClientThrottledError,
  GenericRequestError,
  PleaseWaitFewMinutes,
  RateLimitError,
  SentryBlock,
)
from loguru import logger

from libraries.scrape import get_proxy

silent_error: list[type[Exception]] = [
  ClientConnectionError,
  ClientForbiddenError,
  ClientLoginRequired,
  ClientThrottledError,
  GenericRequestError,
  PleaseWaitFewMinutes,
  RateLimitError,
  SentryBlock
]

client = Client(proxy=get_proxy()) # pyright: ignore[reportArgumentType]

def silent_errors(error: Exception):
  if type(error) in silent_error:
    logger.error(f"Silent error: {error}")
    sleep(60 * 6)
  raise error

