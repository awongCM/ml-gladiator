from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run():
  wine = load_wine()
  X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.25,
    random_state=42,
    stratify=wine.target,
  )

  pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC()),
  ])

  # Linear SVC ignores gamma; search it only for the RBF kernel.
  search = GridSearchCV(
    pipeline,
    param_grid=[
      {
        "classifier__kernel": ["linear"],
        "classifier__C": [0.1, 1, 10],
      },
      {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ["scale", "auto"],
      },
    ],
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
  )
  search.fit(X_train, y_train)
  predictions = search.predict(X_test)

  print("Hyperparameter tuning: Wine classification with GridSearchCV")
  print(f"Best params: {search.best_params_}")
  print(f"Best CV score: {search.best_score_:.3f}")
  print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
  print(classification_report(y_test, predictions, target_names=wine.target_names))
