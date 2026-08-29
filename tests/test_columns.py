import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.utils import Bunch

ROOT = Path(__file__).resolve().parents[1]


def load_columns():
  spec = importlib.util.spec_from_file_location(
    "column_transformer_learning",
    ROOT / "column-transformer-learning.py",
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def fake_adult():
  data = pd.DataFrame({
    "age": [20, 40] * 20,
    "hours-per-week": [40] * 40,
    "workclass": ["Private"] * 38 + [None, None],
    "education": ["Bachelors"] * 40,
    "marital-status": ["Never-married"] * 20 + ["Married-civ-spouse"] * 20,
    "occupation": ["Sales"] * 38 + [None, None],
  })
  target = pd.Series([">50K", "<=50K"] * 20)
  return Bunch(data=data, target=target)


class ColumnTransformerTutorialTests(unittest.TestCase):
  def test_run_fits_with_mocked_openml_and_nans(self):
    module = load_columns()
    with patch.object(module, "fetch_openml", return_value=fake_adult()):
      buf = io.StringIO()
      with redirect_stdout(buf):
        module.run()
    out = buf.getvalue()
    self.assertIn("Accuracy:", out)
    self.assertIn("Transformed features:", out)
    self.assertIn("Missing values:", out)
    self.assertNotIn("skipped", out.lower())

  def test_run_skips_when_fetch_fails(self):
    module = load_columns()
    with patch.object(module, "fetch_openml", side_effect=OSError("offline")):
      buf = io.StringIO()
      with redirect_stdout(buf):
        module.run()
    self.assertIn("skipped", buf.getvalue().lower())

  def test_accepts_dotted_positive_label(self):
    module = load_columns()
    adult = fake_adult()
    adult.target = pd.Series([">50K.", "<=50K"] * 20)
    with patch.object(module, "fetch_openml", return_value=adult):
      buf = io.StringIO()
      with redirect_stdout(buf):
        module.run()
    self.assertIn("Accuracy:", buf.getvalue())


if __name__ == "__main__":
  unittest.main()
