from ..libraries.scrape import read

df = read("user.pkl")
df = df.unique(["uid"])
df.write_avro("user.avro")

