import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_main():
  spec = importlib.util.spec_from_file_location("gladiator_main", ROOT / "main.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class ModuleRegistryTests(unittest.TestCase):
  def test_help_lists_registered_modules(self):
    result = subprocess.run(
      [sys.executable, str(ROOT / "main.py"), "--help"],
      check=True,
      capture_output=True,
      text=True,
    )
    for name in ("statistical", "text", "tuning", "ensemble", "unsupervised"):
      with self.subTest(name=name):
        self.assertIn(name, result.stdout)

  def test_modules_map(self):
    app = load_main()
    self.assertEqual(
      set(app.MODULES),
      {"statistical", "text", "tuning", "ensemble", "unsupervised"},
    )


if __name__ == "__main__":
  unittest.main()
