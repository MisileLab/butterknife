from time import sleep
from secrets import SystemRandom
from pathlib import Path
from re import compile, sub

from httpx import ConnectTimeout
from loguru import logger
from twscrape import Tweet, gather # pyright: ignore[reportMissingTypeStubs]
from twscrape.logger import set_log_level # pyright: ignore[reportMissingTypeStubs]

from libraries.scrape import Data, Provider, User, append, is_unique_user, read, api

url_filter = compile(r"(https?:\/\/)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

def get_value[T](v: T | None) -> T:
  if v is None:
    raise TypeError()
  return v

async def get_tweets(user_id: int) -> list[Tweet]:
  r = SystemRandom().randint(1, 10)
  logger.info(f"sleep {r} sec")
  sleep(r)
  return await gather(api.user_tweets(user_id)) # pyright: ignore[reportUnknownMemberType]

async def get_user(userid: int) -> int:
  r = SystemRandom().randint(1, 10)
  logger.info(f"sleep {r} sec")
  sleep(r)
  res = await api.user_by_id(userid) # pyright: ignore[reportUnknownMemberType]
  if res is None:
    raise TypeError("user is None")
  return res.id

set_log_level("DEBUG")
if not Path("./results").is_dir():
  Path("./results").mkdir()

async def main():
  df = read("data.avro")
  df_user = read("user.avro")

  for _i in df_user.to_dicts():
    i: User = User.model_validate(_i)
    uid = int(i.uid)
    logger.debug(uid)
    if not is_unique_user(df, i.uid, Provider.x):
      logger.info("skip because exists")
      continue
    data: list[str] = []
    nxt_skip = False
    tweets: list[Tweet]
    try:
      tweets = await get_tweets(uid)
    except ConnectTimeout:
      logger.warning("timeout error, try again")
      tweets = await get_tweets(uid)
    for j in tweets:
      if nxt_skip:
        nxt_skip = False
        continue
      if j.retweetedTweet is not None:
        logger.warning("retweeted")
        nxt_skip = True
        continue
      for mention in j.mentionedUsers:
        logger.info(f"delete {mention.username}")
        j.rawContent = j.rawContent.replace(f"@{mention.username}", "").replace(f"@{mention.displayname}", "")
      j.rawContent = sub(url_filter, "", j.rawContent)
      if j.rawContent == "":
        continue
      data.append(j.rawContent)
    if len(data) == 0:
      logger.error(f"no tweets on {uid}, skip it")
      continue
    df = append(df, Data(
      uid=str(uid),
      name=i.name,
      user_type=i.user_type,
      data=data,
      provider=i.provider
    ))

  df.write_avro("data.avro")

