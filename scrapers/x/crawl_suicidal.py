from loguru import logger

from asyncio import run
from time import sleep
from secrets import SystemRandom

from ...libraries.scrape import UserType, is_unique, read, append, User, api

base_query = "site:x.com"
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
      if is_unique(df, "uid", user.id):
        df = append(df, User(
          uid=user.id,
          name=user.username,
          url=user.url,
          user_type=UserType.suicidal
        ))
      r = SystemRandom().randint(0, 10)
      logger.debug(f"sleep {r} secs")
      sleep(r)
  df.write_avro("user.avro")

if __name__ == "__main__":
  run(main())
