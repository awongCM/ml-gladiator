from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
  predictions = model.predict(X_test)

  print("Statistical learning: Iris classification")
  print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
  print(classification_report(y_test, predictions, target_names=iris.target_names))
