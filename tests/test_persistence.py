import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_persistence():
  spec = importlib.util.spec_from_file_location(
    "model_persistence_learning",
    ROOT / "model-persistence-learning.py",
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class PersistenceTutorialTests(unittest.TestCase):
  def test_run_round_trip(self):
    module = load_persistence()
    buf = io.StringIO()
    with redirect_stdout(buf):
      module.run()
    out = buf.getvalue()
    self.assertIn("Predictions match:  True", out)
    self.assertIn("Artifact exists: True", out)
    self.assertIn("pickle", out.lower())
    self.assertIn("removed", out.lower())

  def test_run_fails_closed_on_mismatch(self):
    module = load_persistence()
    real_load = module.joblib.load

    def load_wrong(path):
      loaded = real_load(path)
      n_classes = len(loaded.classes_)
      loaded.predict = lambda X: np.zeros(len(X), dtype=int)
      loaded.predict_proba = lambda X: np.zeros((len(X), n_classes))
      return loaded

    with patch.object(module.joblib, "load", side_effect=load_wrong):
      with self.assertRaises(AssertionError):
        module.run()


if __name__ == "__main__":
  unittest.main()
