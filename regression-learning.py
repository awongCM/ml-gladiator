from urllib.error import URLError

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _evaluate(name, model, X_train, X_test, y_train, y_test, feature_names):
  cv_neg_mse = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
  )
  cv_rmse = np.sqrt(-cv_neg_mse)

  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, predictions))
  r2 = r2_score(y_test, predictions)
  coef = model.named_steps["model"].coef_
  nnz = int(np.sum(np.abs(coef) > 1e-8))

  print(name)
  print(f"  Holdout RMSE: {rmse:.3f}  R2: {r2:.3f}")
  print(f"  5-fold CV RMSE (train): {cv_rmse.mean():.3f} (+/- {cv_rmse.std() * 2:.3f})")
  print(f"  Non-zero coefficients: {nnz}/{len(coef)}")
  for feature, weight in zip(feature_names, coef):
    print(f"    {feature}: {weight:.3f}")


def run():
  try:
    housing = fetch_california_housing()
  except (OSError, URLError) as exc:
    print(
      "Regression tutorial skipped: could not download California housing "
      f"({type(exc).__name__}: {exc})"
    )
    return

  X_train, X_test, y_train, y_test = train_test_split(
    housing.data,
    housing.target,
    test_size=0.25,
    random_state=42,
  )

  print("Regression with regularization: California housing")
  print(f"Samples: {housing.data.shape[0]}, features: {housing.data.shape[1]}")
  print("Target: median house value in units of $100,000")

  _evaluate(
    "LinearRegression (baseline)",
    Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    X_train,
    X_test,
    y_train,
    y_test,
    housing.feature_names,
  )
  _evaluate(
    "Ridge (L2, alpha=1.0)",
    Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    X_train,
    X_test,
    y_train,
    y_test,
    housing.feature_names,
  )
  _evaluate(
    "Lasso (L1, alpha=0.01)",
    Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000))]),
    X_train,
    X_test,
    y_train,
    y_test,
    housing.feature_names,
  )
