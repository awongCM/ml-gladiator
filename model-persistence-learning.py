import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run():
  iris = datasets.load_iris()
  X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.25,
    random_state=42,
    stratify=iris.target,
  )

  model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
  ])
  model.fit(X_train, y_train)
  original_predictions = model.predict(X_test)
  original_proba = model.predict_proba(X_test)

  print("Model persistence: save and reload with joblib")
  print("joblib uses pickle; do not load files from untrusted sources.")

  with tempfile.TemporaryDirectory() as temp_dir:
    model_path = Path(temp_dir) / "iris_pipeline.joblib"
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    loaded_predictions = loaded_model.predict(X_test)
    loaded_proba = loaded_model.predict_proba(X_test)

    print(f"Model path: {model_path}")
    print(f"Artifact exists: {model_path.exists()}")
    print(f"Original accuracy:  {accuracy_score(y_test, original_predictions):.3f}")
    print(f"Reloaded accuracy:  {accuracy_score(y_test, loaded_predictions):.3f}")
    print(f"Predictions match:  {np.array_equal(original_predictions, loaded_predictions)}")
    if not np.array_equal(original_predictions, loaded_predictions):
      raise AssertionError("Reloaded model predictions do not match the original pipeline")
    if not np.allclose(original_proba, loaded_proba):
      raise AssertionError("Reloaded model probabilities do not match the original pipeline")

  print("Temporary artifact directory removed after the round-trip.")
