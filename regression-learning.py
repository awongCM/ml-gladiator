import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _evaluate(name, model, X_train, X_test, y_train, y_test):
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, predictions))
  r2 = r2_score(y_test, predictions)
  print(f"{name}")
  print(f"  RMSE: {rmse:.3f}")
  print(f"  R2:   {r2:.3f}")


def run():
  housing = fetch_california_housing()
  X_train, X_test, y_train, y_test = train_test_split(
    housing.data,
    housing.target,
    test_size=0.25,
    random_state=42,
  )

  print("Regression with regularization: California housing")

  _evaluate(
    "LinearRegression (baseline)",
    Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    X_train,
    X_test,
    y_train,
    y_test,
  )
  _evaluate(
    "Ridge (L2)",
    Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    X_train,
    X_test,
    y_train,
    y_test,
  )
  _evaluate(
    "Lasso (L1)",
    Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000))]),
    X_train,
    X_test,
    y_train,
    y_test,
  )
