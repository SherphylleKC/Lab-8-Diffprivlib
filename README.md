# Assessment Lab 8

### Instructions
**To get started on your Workshop, please follow the below instructions.** <br>

Diffprivlib is a general-purpose library for experimenting with, investigating and developing applications in, differential privacy. <br>

Use Diffprivlib if you are looking to:
- Experiment with differential privacy.
- Explore the impact of differential privacy on machine learning accuracy using 
classification and clustering models.
- Build your own differential privacy applications, using our extensive collection of 
mechanisms. <br>

In this workshop, we will experience Diffprivlib v0.4. You can follow the below steps and 
codes to implement your environment in ***Python***.

Step 1: Install the following libraries by using the following commands to avoid any errors:
- *Diffprivlib*; command to install is `pip install diffprivlib`.
- *Matplotlib*; command to install is `pip install matplotlib`.

**Step 2**: Download Iris dataset
<br>
**Step 3**:  Train a differentially private naive Bayes classifier. Our classifier runs just like a `sklearn` classifier, so you can get up and running quickly. The privacy level is controlled by the parameter `epsilon`, which is passed to the classifier at initialisation (e.g. `GaussianNB(epsilon=0.1)`). The default is `epsilon = 1.0`.
```
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
```
<br>
**Step 4**: We can now classify unseen examples, knowing that the trained model is differentially private and preserves the privacy of the 'individuals' in the training set.
<br>
**Step 5**: Every time the model is trained with `.fit()`, a different model is produced due to the randomness of differential privacy. The accuracy will therefore change, even if it's re-trained with the same training data.
<br>
**Step 6**: We can easily evaluate the accuracy of the model for various `epsilon` values and plot it with `matplotlib`.



