from pathlib import Path
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
accounts: list[tuple[str, str]] = []
len_accounts = len(accounts)
account_index = 0
for i in Path("accounts.txt").read_text().split("\n"):
  if i.strip() == "":
    continue
  data = i.split(":")
  accounts.append((i[0], i[1]))

def silent_errors(error: Exception):
  if type(error) in silent_error:
    logger.error(f"Silent error: {error}")
    login_resp = False
    while not login_resp:
      global account_index
      account_index += 1
      if account_index == len_accounts:
        account_index = 0
      login_resp = client.login(accounts[account_index][0], accounts[account_index][1])
    logger.info(f"login to {accounts[account_index][0]}")
    return
  raise error

