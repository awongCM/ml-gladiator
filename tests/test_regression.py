import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.utils import Bunch

ROOT = Path(__file__).resolve().parents[1]


def load_regression():
  spec = importlib.util.spec_from_file_location(
    "regression_learning",
    ROOT / "regression-learning.py",
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def fake_housing():
  rng = np.random.RandomState(0)
  data = rng.randn(80, 4)
  target = data @ np.array([2.0, -1.0, 0.0, 0.5]) + rng.randn(80) * 0.05
  return Bunch(data=data, target=target, feature_names=list("ABCD"))


class RegressionTutorialTests(unittest.TestCase):
  def test_run_prints_metrics_with_mocked_fetch(self):
    module = load_regression()
    with patch.object(module, "fetch_california_housing", return_value=fake_housing()):
      buf = io.StringIO()
      with redirect_stdout(buf):
        module.run()
    out = buf.getvalue()
    self.assertIn("Holdout RMSE", out)
    self.assertIn("5-fold CV RMSE", out)
    self.assertIn("Non-zero coefficients", out)
    self.assertIn("LinearRegression", out)
    self.assertIn("Ridge (L2, alpha=1.0)", out)
    self.assertIn("Lasso (L1, alpha=0.01)", out)

  def test_run_skips_when_fetch_fails(self):
    module = load_regression()
    with patch.object(module, "fetch_california_housing", side_effect=OSError("offline")):
      buf = io.StringIO()
      with redirect_stdout(buf):
        module.run()
    self.assertIn("skipped", buf.getvalue().lower())


if __name__ == "__main__":
  unittest.main()
