# Machine Learning in Gladiator mode

My so-called machine learning tutorial marathon using Python and Scikit-Learn tools.

Python Libraries used.
* Numpy
* Sklearn
* Scipy
* Pandas
* Joblib

## Exercises

| Module | Command | Topics |
|--------|---------|--------|
| Statistical learning | `python3 main.py statistical` | Pipelines, scaling, logistic regression (Iris) |
| Text learning | `python3 main.py text` | TF-IDF, naive Bayes (20 Newsgroups) |
| Hyperparameter tuning | `python3 main.py tuning` | GridSearchCV, SVM, cross-validation (Wine) |
| Ensemble learning | `python3 main.py ensemble` | RandomForest, feature importances (Breast cancer) |
| Unsupervised learning | `python3 main.py unsupervised` | PCA, KMeans, clustering metrics (Wine) |
| Regression with regularization | `python3 main.py regression` | Pipeline, StandardScaler, LinearRegression vs Ridge vs Lasso, RMSE/R² (California housing) |
| ColumnTransformer | `python3 main.py columns` | Nested numeric/categorical pipelines, imputation, OneHotEncoder, LogisticRegression (Adult income) |
| Model persistence | `python3 main.py persistence` | Save/load pipelines with joblib (Iris) |

Run all exercises (`python3 main.py all` is the same):

```bash
python3 main.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```
