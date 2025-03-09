from loguru import logger

from time import sleep
from secrets import SystemRandom

from libraries.scrape import UserType, is_unique, read, append, User
from .lib import client

suicidals = [
  "자해", "자해러", "자해계",
  "자살", "자살시도", "자살충동", "자살사고",
  "죽고싶다", "죽고싶어"
]

def unwrap(v: str | None) -> str:
  if v is None:
    raise TypeError()
  return v

async def main():
  df = read("user.avro")

  for suicidal_tag in suicidals:
    logger.info(suicidal_tag)
    data = client.hashtag_medias_top(suicidal_tag, amount=30)
    data_recent = client.hashtag_medias_recent(suicidal_tag, amount=30)
    data.extend(data_recent)
    for post in data:
      user = post.user
      logger.info(user.username)
      if is_unique(df, "uid", user.pk):
        df = append(df, User(
          uid=user.pk,
          name=unwrap(user.username),
          url="",
          user_type=UserType.suicidal
        ))
      r = SystemRandom().randint(0, 10)
      logger.debug(f"sleep {r} secs")
      sleep(r)
  df.write_avro("user.avro")

