import argparse
import importlib.util
from pathlib import Path


def _load_module(name, filename):
  path = Path(__file__).with_name(filename)
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


statistical_learning = _load_module("statistical_learning", "statistical-learning.py")
text_learning = _load_module("text_learning", "text-learning.py")


def main():
  parser = argparse.ArgumentParser(
    description="Machine Learning Gladiator tutorial exercises",
  )
  parser.add_argument(
    "module",
    nargs="?",
    choices=["statistical", "text", "all"],
    default="all",
    help="which exercise to run (default: all)",
  )
  args = parser.parse_args()

  print("ML Goodness begins here\n")

  if args.module in ("statistical", "all"):
    statistical_learning.run()
    if args.module == "all":
      print()

  if args.module in ("text", "all"):
    text_learning.run()


if __name__ == "__main__":
  main()
