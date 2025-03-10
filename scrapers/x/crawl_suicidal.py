from loguru import logger

from time import sleep
from secrets import SystemRandom

from libraries.scrape import Provider, UserType, is_unique, read, append, User, api

suicidals = [
  "자해", "자해러", "자해계", "자해흉터", "자해글귀", "자해하는사람은나쁜사람이아닙니다",
  "자살", "자살시도", "자살충동", "자살사고",
  "죽고싶다", "죽고싶어"
]


async def main():
  df = read("user.avro")

  for suicidal_tag in suicidals:
    logger.info(suicidal_tag)
    async for tweet in api.search(f"#{suicidal_tag}"): # pyright: ignore[reportUnknownMemberType]
      user = tweet.user
      logger.info(user.username)
      userid = str(user.id)
      if is_unique(df, "uid", userid):
        df = append(df, User(
          uid=userid,
          name=user.username,
          user_type=UserType.suicidal,
          provider=Provider.x
        ))
      r = SystemRandom().randint(0, 10)
      logger.debug(f"sleep {r} secs")
      sleep(r)
  df.write_avro("user.avro")

