
# coding: utf-8

# Author: Pradeep K.Pant

# # Natural Language Processing Project
# 
# Welcome to the NLP Project for this section of the course. In this NLP project you will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews. This will be a simpler procedure than the lecture, since we will utilize the pipeline methods for more complex tasks.
# 
# We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).
# 
# Each observation in this dataset is a review of a particular business by a particular user.
# 
# The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# 
# The "cool" column is the number of "cool" votes this review received from other Yelp users. 
# 
# All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.
# 
# The "useful" and "funny" columns are similar to the "cool" column.
# 
# Let's get started! Just follow the directions below!

# Imports essential libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# This is needed to show the plot in Jupyter notebook
get_ipython().magic('matplotlib inline')

# The Data
# 
# Read the yelp.csv file and set it as a dataframe called yelp.

yelp = pd.read_csv('yelp.csv')


# Check the head, info , and describe methods on yelp.

yelp.head()

yelp.info()

yelp.describe()

# Create a new column called "text length" which is the number of words in the text column.

yelp['text length'] = yelp['text'].apply(len)


# Lets do some exploratory data analysis
# 
# Let's explore the data
# 
# Imports
# 
# Import the data visualization libraries if you haven't done so already.

sns.set_style('white')

# Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length', bins=50)


# Create a boxplot of text length for each star category.

sns.boxplot(x='stars',y='text length',data=yelp, palette='rainbow')


# Create a countplot of the number of occurrences for each type of star rating.

sns.countplot(x='stars',data=yelp, palette='rainbow')

# Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:

stars = yelp.groupby('stars').mean()
stars


# To check the correlation between values, use the corr() method on that groupby dataframe to produce this dataframe:

stars.corr()

# Then use seaborn to create a heatmap based off that .corr() dataframe:

sns.heatmap(stars.corr(),annot=True)

# NLP Classification Task
# Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.
# 
# Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.

yelp_class = yelp[(yelp['stars']==1)| (yelp['stars']==5)]
yelp_class.info

# Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)

X = yelp_class['text']
y = yelp_class['stars']

# Import CountVectorizer and create a CountVectorizer object.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.

X = cv.fit_transform(X)

# Train Test Split
# 
# Let's split our data into training and testing data.
# 
# Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 
# Note: One can use any value for random state and test size

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# Training a Model
 
# Time to train a model!
# 
# Import MultinomialNB and create an instance of the estimator and call is nb 

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Now fit nb using the training data.

nb.fit(X_train,y_train)


# Predictions and Evaluations
# 
# Time to see how our model did!
# 
# Use the predict method off of nb to predict labels from X_test.

predictions = nb.predict(X_test)


# Create a confusion matrix and classification report using these predictions and y_test

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,predictions))
print('\n')
print (classification_report(y_test,predictions))

# Now, Let's see what happens if we try to include TF-IDF to this process using a pipeline.**

# Using Text Processing
# Import TfidfTransformer from sklearn. 

from sklearn.feature_extraction.text import TfidfTransformer

# Import Pipeline from sklearn. **
from sklearn.pipeline import Pipeline

# Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()

pipe = Pipeline([('bow',CountVectorizer()),('tfidf',TfidfTransformer()),('model',MultinomialNB())])


# Using the Pipeline
# 
# Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the text

# ### Train Test Split
# 
# Redo the train test split on the yelp_class object.

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels

pipe.fit(X_train,y_train)

# Predictions and Evaluation
# 
#  Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.

predictions = pipe.predict(X_test)

# Lets check again confusion matrix and classification report
print(confusion_matrix(y_test,predictions))
print('\n')
print (classification_report(y_test,predictions))

# Looks like Tf-Idf actually made things worse! That is it for this project. 

# Some other things to try... 
# Try going back and playing around with the pipeline steps and seeing if creating a custom analyzer helps Or recreate the pipeline with just the CountVectorizer() and NaiveBayes. Does changing the ML model at the end to another classifier help at all?