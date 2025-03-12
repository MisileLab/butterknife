from time import sleep
from secrets import SystemRandom

from loguru import logger
from pydantic import BaseModel

from .lib import client
from libraries.scrape import Provider, UserType, is_unique_user, read, append, User

suicidals = [
  "자해", "자해러", "자해계", "자해흉터", "자해글귀", "자해하는사람은나쁜사람이아닙니다",
  "자살", "자살시도", "자살충동", "자살사고",
  "죽고싶다", "죽고싶어"
]

class InternalUser(BaseModel):
  pk: str | int
  id: str | int | None
  username: str | None

async def main():
  df = read("user.avro")

  for suicidal_tag in suicidals:
    logger.info(suicidal_tag)
    data = client.hashtag_medias_top(suicidal_tag, count=30) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    data.extend(client.hashtag_medias_recent(suicidal_tag, count=30)) # pyright: ignore[reportUnknownMemberType]
    for post in data: # pyright: ignore[reportUnknownVariableType]
      user = InternalUser.model_validate(post["user"])
      logger.info(user.username)
      userid = str(user.pk)
      if is_unique_user(df, userid, Provider.instagram):
        df = append(df, User(
          uid=userid,
          name="" if user.username is None else user.username,
          user_type=UserType.suicidal,
          provider=Provider.instagram
        ))
      r = SystemRandom().randint(0, 10)
      logger.debug(f"sleep {r} secs")
      sleep(r)
  df.write_avro("user.avro")

