from asyncio import run
from importlib import import_module

if __name__ == '__main__':
  print("this is boilerplate because can't import correctly")
  run(import_module(input()).main()) # pyright: ignore[reportAny]

