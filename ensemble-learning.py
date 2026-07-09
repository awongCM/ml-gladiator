from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split


def run():
  cancer = load_breast_cancer()
  X_train, X_test, y_train, y_test = train_test_split(
    cancer.data,
    cancer.target,
    test_size=0.25,
    random_state=42,
    stratify=cancer.target,
  )

  model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
  )
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)

  cv_scores = cross_val_score(model, cancer.data, cancer.target, cv=5, scoring="accuracy")

  top_features = sorted(
    zip(cancer.feature_names, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
  )[:5]

  print("Ensemble learning: Breast cancer with RandomForest")
  print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
  print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
  print(classification_report(y_test, predictions, target_names=cancer.target_names))
  print("Top feature importances:")
  for name, importance in top_features:
    print(f"  {name}: {importance:.3f}")
