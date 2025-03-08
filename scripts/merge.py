from polars import col
from ..libraries.scrape import read

df = read("data.avro")
df_user = df.filter(col("confirmed"))
df_user = df.drop(["data", "confirmed"])
df_user.write_avro("user.avro")

