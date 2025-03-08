from libraries.scrape import get_inactives, get_usernames, api
from asyncio import run

async def main():
  await api.pool.relogin(await get_usernames())

async def relogin_force():
  await api.pool.relogin(await get_inactives())

if __name__ == "__main__":
  run(main())
