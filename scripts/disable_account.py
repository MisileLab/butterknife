from ..libraries.scrape import api

async def main():
  acc_name = input().strip()
  await api.pool.set_active(acc_name, False)

