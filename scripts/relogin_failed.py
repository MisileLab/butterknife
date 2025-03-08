from ..libraries.scrape import api

async def main():
  await api.pool.relogin_failed()

