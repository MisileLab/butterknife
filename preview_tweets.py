from contextlib import suppress
from copy import deepcopy

from polars import col
from pypager.pager import Pager # pyright: ignore[reportMissingTypeStubs]
from pypager.source import StringSource # pyright: ignore[reportMissingTypeStubs]

from libraries.scrape import Data, UserType, read

data = read("data.avro")
data_res = deepcopy(data)
mapping = {
  "y": UserType.suicidal,
  "n": UserType.normal,
  "r": UserType.ignored
}

with suppress(KeyboardInterrupt):
  for index, _i in enumerate(data.select(col("confirmed") is False).to_dicts()):
    i = Data.model_validate(_i)
    tweets: list[str] = i.data
    suicidal_comments: str = "\n--sep--\n".join(i for i in tweets if i.count("자살")+i.count("자해") != 0)
    if suicidal_comments != "":
      p = Pager()
      _ = p.add_source(StringSource(suicidal_comments))
      p.run()
    elif i.user_type == UserType.suicidal:
      print("previously suicidal but none found")
    else:
      print("suicidal none found")
    full = suicidal_comments == "" or input("do you want to see full? [y/n]: ").lower() == "y"
    if full:
      p = Pager()
      _ = p.add_source(StringSource("\n--sep--\n".join(i for i in tweets)))
      p.run()
    user_type = input("is this suicidal? (if not normal message and it is something like news, input 'r') [y/n/r]: ")
    while user_type.lower() not in ["y", "n", "r"]:
      user_type = input("is this suicidal? (if not normal message and it is something like news, input 'r') [y/n/r]: ")
    data_res[index, 'user_type'] = mapping[user_type.lower()]
    data_res[index, 'confirmed'] = True

data_res.write_avro("data.avro")
