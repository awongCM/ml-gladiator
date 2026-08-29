from urllib.error import URLError

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# A mixed-type slice of Adult, not the full feature set.
NUMERIC_FEATURES = ["age", "hours-per-week"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital-status", "occupation"]


def run():
  try:
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
  except (OSError, URLError, ValueError) as exc:
    print(
      "ColumnTransformer tutorial skipped: could not download Adult from OpenML "
      f"({type(exc).__name__}: {exc})"
    )
    return

  features = adult.data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
  labels = adult.target.astype(str).str.strip()
  target = labels.isin([">50K", ">50K."]).astype(int)

  missing = features.isna().sum()
  missing_nonzero = {name: int(count) for name, count in missing.items() if count}
  print("ColumnTransformer: Adult income with mixed feature types")
  print(f"Missing values: {int(missing.sum())} ({missing_nonzero})")

  X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42,
    stratify=target,
  )

  numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
  ])
  categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
  ])
  preprocessor = ColumnTransformer(
    transformers=[
      ("num", numeric_pipeline, NUMERIC_FEATURES),
      ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
  )

  model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
  ])
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)

  fitted = model.named_steps["preprocessor"]
  n_transformed = fitted.transform(X_train).shape[1]
  print(f"Raw features: {features.shape[1]}")
  print(f"Transformed features: {n_transformed}")
  print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
  print(classification_report(y_test, predictions, target_names=["<=50K", ">50K"]))
