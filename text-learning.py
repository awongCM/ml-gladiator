from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


CATEGORIES = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]


def run():
  data = fetch_20newsgroups(subset="train", categories=CATEGORIES, shuffle=True, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=42,
    stratify=data.target,
  )

  model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", MultinomialNB()),
  ])
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)

  print("Text learning: 20 Newsgroups classification")
  print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
  print(classification_report(y_test, predictions, target_names=data.target_names))
