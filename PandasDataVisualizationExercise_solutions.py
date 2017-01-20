
# coding: utf-8
# Solutions -- code for Pandas built in functions 
# # Pandas Data Visualization Exercise
# This is just a quick exercise for you to review the various plots we showed earlier. Use **df3** to replicate the following plots. 
# Preparation
import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')
# to show the plot in juypter notebook

get_ipython().magic('matplotlib inline')
df3.info()
# Check top of the head of the data frame
df3.head()

# Q1: ** Recreate this scatter plot of b vs a. Note the color and size of the points. Also note the figure size. See if you can figure out how to stretch it in a similar fashion. Remeber back to your matplotlib lecture...**

df3.plot.scatter(x='a',y='b',figsize=(12,3),s=50,c='red')

# Q1:** Create a histogram of the 'a' column.**

df3['a'].hist()

# ** These plots are okay, but they don't look very polished. Use style sheets to set the style to 'ggplot' and redo the histogram from above. Also figure out how to add more bins to it.***

plt.style.use('ggplot')
df3['a'].plot.hist(bins=20,alpha=0.5)
# ** Create a boxplot comparing the a and b columns.**
df['a','b'].plot.box()

# ** Create a kde plot of the 'd' column **
df3['d'].plot.kde()
# adding line width and line type 
df3['d'].plot.kde(lw=5,ls='--')


# ** Figure out how to increase the linewidth and make the linestyle dashed. (Note: You would usually not dash a kde plot line)**

# ** Create an area plot of all the columns for just the rows up to 30. (hint: use .ix).**
df3.ix[0:30].plot.area(alpha=0.4)
f = plt.figure()
df3.ix[0:30].plot.area(alpha=0.4)
# Now to put the legend in such a away so that ot doesn't overlap the actual plot we'll use loc parameter with legend
plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
plt.show()

# ** Try searching Google for a good stackoverflow link on this topic. If you can't find it on your own - [use this one for a hint.](http://stackoverflow.com/questions/23556153/how-to-put-legend-outside-the-plot-with-pandas)**

# END 