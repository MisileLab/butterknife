from dataclasses import dataclass
from pathlib import Path
from secrets import SystemRandom
from time import sleep

from instagrapi import Client # pyright: ignore[reportMissingTypeStubs]
from instagrapi.exceptions import ( # pyright: ignore[reportMissingTypeStubs]
  ClientConnectionError,
  ClientForbiddenError,
  ClientLoginRequired,
  ClientThrottledError,
  GenericRequestError,
  PleaseWaitFewMinutes,
  ProxyAddressIsBlocked,
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

@dataclass
class Initalizer:
  account_index = 0

client = Client(proxy=get_proxy()) # pyright: ignore[reportArgumentType]
accounts: list[tuple[str, str]] = []
len_accounts = len(accounts)
initalizer = Initalizer()
for i in Path("accounts.txt").read_text().split("\n"):
  if i.strip() == "":
    continue
  data = i.split(":")
  accounts.append((i[0], i[1]))
_ = client.login(accounts[initalizer.account_index][0], accounts[initalizer.account_index][1])
initalizer.account_index += 1

def silent_errors(error: Exception):
  if isinstance(error, ProxyAddressIsBlocked):
    r = SystemRandom().randint(5, 10)
    logger.error(f"proxy blocked, sleep {r} mins")
    sleep(r * 60)
    return
  if type(error) in silent_error:
    logger.error(f"Silent error: {error}")
    login_resp = False
    org_accounts_index = initalizer.account_index
    while not login_resp and org_accounts_index != initalizer.account_index:
      initalizer.account_index += 1
      if initalizer.account_index == len_accounts:
        initalizer.account_index = 0
      login_resp = client.login(
        accounts[initalizer.account_index][0],
        accounts[initalizer.account_index][1]
      )
    if login_resp:
      logger.info(f"login to {accounts[initalizer.account_index][0]}")
    else:
      logger.error("all accounts are blocked")
      raise error
    return
  raise error

