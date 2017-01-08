
# coding: utf-8
# Solutions: Code for numpy library exercises
# Pradeep K. Pant @2017
# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # NumPy Exercises 
# 
# Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks, and then you'll be asked some more complicated questions.

# #### Import NumPy as np

# In[2]:

import numpy as np


# #### Create an array of 10 zeros 

# In[3]:

np.zeros(10)


# In[ ]:




# In[2]:




# In[ ]:




# In[ ]:




# #### Create an array of 10 ones

# In[4]:

np.ones(10)


# #### Create an array of 10 fives

# In[8]:

newarr = np.ones(10)


# In[9]:

newarr*5


# In[4]:




# #### Create an array of the integers from 10 to 50

# In[14]:

np.arange(10,51)


# In[5]:




# #### Create an array of all the even integers from 10 to 50

# In[16]:

np.arange(10,52,2)


# In[6]:




# #### Create a 3x3 matrix with values ranging from 0 to 8

# In[71]:

np.arange(9).reshape(3,3)


# In[7]:




# #### Create a 3x3 identity matrix

# In[18]:

np.eye(3)


# #### Use NumPy to generate a random number between 0 and 1

# In[31]:

np.random.rand(1)


# In[15]:




# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# In[32]:

np.random.randn(25)


# In[33]:




# #### Create the following matrix:

# In[72]:

np.linspace(0.01,1,100).reshape(10,10)


# In[ ]:




# 

# In[35]:




# #### Create an array of 20 linearly spaced points between 0 and 1:

# In[40]:

np.linspace(0,1,20)


# In[36]:




# ## Numpy Indexing and Selection
# 
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# In[41]:

mat = np.arange(1,26).reshape(5,5)
mat


# In[73]:

mat[2:,1:]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[29]:

# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[75]:

mat[3,4]


# In[ ]:




# In[ ]:




# In[30]:




# In[ ]:




# In[77]:

mat[:3,1:2]


# In[79]:

mat[4,:]


# In[46]:




# In[32]:

# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[80]:

mat[3:5,:]


# ### Now do the following

# #### Get the sum of all the values in mat

# In[81]:

mat


# In[82]:

np.sum(mat)


# #### Get the standard deviation of the values in mat

# In[83]:

np.std(mat)


# #### Get the sum of all the columns in mat

# In[84]:

mat


# In[85]:

mat.sum(axis=0)


# # Great Job!
