from polars import DataFrame, read_avro

from libraries.scrape import append

filename = input("file: ")
df = read_avro(filename)
migrate_df = DataFrame()

for i in df.to_dicts():
  i['user_type'] = 1 if i['suicidal'] else 0
  del i['suicidal']
  migrate_df = append(migrate_df, i)

migrate_df.write_avro(filename)

