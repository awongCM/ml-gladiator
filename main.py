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
hyperparameter_tuning = _load_module("hyperparameter_tuning", "hyperparameter-tuning.py")
ensemble_learning = _load_module("ensemble_learning", "ensemble-learning.py")
unsupervised_learning = _load_module("unsupervised_learning", "unsupervised-learning.py")

MODULES = {
  "statistical": ("Statistical learning", statistical_learning),
  "text": ("Text learning", text_learning),
  "tuning": ("Hyperparameter tuning", hyperparameter_tuning),
  "ensemble": ("Ensemble learning", ensemble_learning),
  "unsupervised": ("Unsupervised learning", unsupervised_learning),
}


def main():
  parser = argparse.ArgumentParser(
    description="Machine Learning Gladiator tutorial exercises",
  )
  parser.add_argument(
    "module",
    nargs="?",
    choices=["all", *MODULES.keys()],
    default="all",
    help="which exercise to run (default: all)",
  )
  args = parser.parse_args()

  print("ML Goodness begins here\n")

  selected = list(MODULES.keys()) if args.module == "all" else [args.module]
  for index, key in enumerate(selected):
    title, module = MODULES[key]
    print(title)
    module.run()
    if index < len(selected) - 1:
      print()


if __name__ == "__main__":
  main()
