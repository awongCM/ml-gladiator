import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "hours-per-week"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital-status", "occupation"]


def run():
  adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
  features = adult.data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
  target = (adult.target == ">50K").astype(int)

  X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42,
    stratify=target,
  )

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", StandardScaler(), NUMERIC_FEATURES),
      ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ],
  )

  model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
  ])
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)

  transformed = preprocessor.fit_transform(X_train)
  print("ColumnTransformer: Adult income with mixed feature types")
  print(f"Raw features: {features.shape[1]}")
  print(f"Transformed features: {transformed.shape[1]}")
  print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
  print(classification_report(y_test, predictions, target_names=["<=50K", ">50K"]))
