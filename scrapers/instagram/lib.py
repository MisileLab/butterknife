from hikerapi import Client # pyright: ignore[reportMissingTypeStubs]

from os import getenv

client = Client(getenv("HIKERAPI_KEY"))

