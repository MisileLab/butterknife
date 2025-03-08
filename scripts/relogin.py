from libraries.scrape import get_inactives, get_usernames, api

async def main():
  await api.pool.relogin(await get_usernames())

async def relogin_force():
  await api.pool.relogin(await get_inactives())

