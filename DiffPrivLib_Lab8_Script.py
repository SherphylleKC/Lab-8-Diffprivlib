from sklearn import datasets
from sklearn.model_selection import train_test_split
from diffprivlib.models import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset.
dataset = datasets.load_iris()

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

# Initialise the GuassianNB classifier with default epsilon.
clf = GaussianNB()          # Default privacy level is <epsilon=1.0> - placed inside the brackets.
clf.fit(X_train, y_train)       # Fit the model with training data
clf.predict(X_test)         # Fit the model with training data

# Print the accuracy of the model on the test set.
print("Test accuracy: %f" %clf.score(X_test, y_test))

# Define a range of epsilon values.
epsilons = np.logspace(-2, 2, 50)                           # You can change the epsilon value of 50 to a value of your choosing to get a different result.
bounds = ([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5])
accuracy = list()

# Evaluate accuracy for different epsilon values.
for epsilon in epsilons:
    clf = GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train, y_train)

    accuracy.append(clf.score(X_test, y_test))

# Plot the results.
plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()