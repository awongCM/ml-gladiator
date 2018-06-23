# Starting Point of Machine Learning Gladiator

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

def main():
  print("ML Goodness begins here")

  print(digits.data)

if __name__ == '__main__':
  main()