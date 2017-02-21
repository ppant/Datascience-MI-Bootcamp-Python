
# coding: utf-8
# Author: Pradeep K. Pant

# K Means Clustering Project 
# 
# For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.
# 
# It is very important to note, we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm. 
# 
# When using the Kmeans algorithm under normal circumstances, it is because you don't have labels. In this case we will use the labels to try to get an idea of how well the algorithm performed, but you won't usually do this for Kmeans, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting!.

# The Data
# 
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Get the Data

# Read in the College_Data file using read_csv. We'll set the first column as the index.

df = pd.read_csv("College_Data",index_col=0)


# Check the head of the data

df.head()


# Check the info() and describe() methods on the data.

df.info()


df.describe()


# It's time to create some data visualizations!
# 
# Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. 

sns.lmplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private',fit_reg=False,palette='coolwarm',size=6,aspect=1)


# Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.

sns.lmplot(x='Outstate',y='F.Undergrad',data=df,hue='Private',fit_reg=False,size=6,aspect=1)

# Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **

g = sns.FacetGrid(df,hue='Private',palette='coolwarm',size=6,aspect=1)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# Create a similar histogram for the Grad.Rate column.**

g = sns.FacetGrid(df,hue='Private',palette='coolwarm',size=6,aspect=1)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?

df[df['Grad.Rate']>100]

# Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.

df['Grad.Rate']['Cazenovia College'] = 100

df[df['Grad.Rate']>100]

g = sns.FacetGrid(df,hue='Private',palette='coolwarm',size=6,aspect=1)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

# K Means Cluster Creation
# Now it is time to create the Cluster labels!
# 
# Import KMeans from SciKit Learn.

from sklearn.cluster import KMeans

# Create an instance of a K Means model with 2 clusters.
kmeans = KMeans(n_clusters=2)

# Fit the model to all the data except for the Private label.

kmeans.fit(df.drop('Private',axis=1))

# What are the cluster center vectors?

kmeans.cluster_centers_

# Evaluation
# 
# There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
# 
# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.

def conveter(private):
    if private == 'Yes':
        return 1
    else:    
        return 0

df['Cluster'] = df['Private'].apply(conveter)

df.head()

# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.

from sklearn.metrics import confusion_matrix, classification_report
print (confusion_matrix(df['Cluster'],kmeans.labels_))
print("\n")
print (classification_report(df['Cluster'],kmeans.labels_))


# Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! Hopefully you can begin to see how K Means is useful for clustering un-labeled data!
