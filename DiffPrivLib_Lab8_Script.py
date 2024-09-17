from sklearn import datasets
from sklearn.model_selection import train_test_split
from diffprivlib.models import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

clf = GaussianNB()
clf.fit(X_train, y_train)
clf.predict(X_test)

print("Test accuracy: %f" %clf.score(X_test, y_test))

epsilons = np.logspace(-2, 2, 50)
bounds = ([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5])
accuracy = list()

for epsilon in epsilons:
    clf = GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train, y_train)

    accuracy.append(clf.score(X_test, y_test))

plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()