from polars import DataFrame, read_avro

from libraries.scrape import Provider, append

filename = input("file: ")
df = read_avro(filename)
migrate_df = DataFrame()

for i in df.to_dicts():
  del i['url']
  i['uid'] = str(i['uid']) # pyright: ignore[reportAny]
  i['provider'] = Provider.x.value
  migrate_df = append(migrate_df, i)

migrate_df.write_avro(filename)

