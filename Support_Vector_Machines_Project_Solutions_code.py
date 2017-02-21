
# coding: utf-8

# Pradeep K. Pant
# Support Vector Machines Project 
 
# ## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
 
# Here's a picture of the three different Iris types:

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 
# ## Get the data

import seaborn as sns
iris = sns.load_dataset('iris')

# Let's visualize the data and get you started!
# Exploratory Data Analysis
 
# Import libraries needed

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

# Create a pairplot of the data set. Which flower species seems to be the most separable?

sns.pairplot(iris,hue='species',palette="Dark2")

# Create a kde plot of sepal_length versus sepal width for setosa species of flower.

setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)


# Train Test Split
# Split your data into a training set and a testing set.

from sklearn.cross_validation import train_test_split
# Make a X and Y parameters
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# Call the SVC() model from sklearn and fit the model to the training data.

from sklearn.svm import SVC

# Instantiate SVC object
svc_model = SVC()
svc_model.fit(X_train,y_train)

# Model Evaluation
# Now get predictions from the model and create a confusion matrix and a classification report.

predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print (confusion_matrix(y_test,predictions))

print (classification_report(y_test,predictions))


# Looking into confusion matrix and classification reports it seems that model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

# Gridsearch Practice
# 
# Import GridsearchCV from SciKit Learn.

from sklearn.grid_search import GridSearchCV


# Create a dictionary called param_grid and fill out some parameters for C and gamma.

param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0,0.01,0.001]}

# Create a GridSearchCV object and fit it to the training data.

grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)


# Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?

grid_predictions = grid.predict(X_test)

print (confusion_matrix(y_test,grid_predictions))

print (classification_report(y_test,grid_predictions))

# We can see now that there is a slight imporvements in results using grid search method though data set is very small. For larger dataset one might see significant change in results.