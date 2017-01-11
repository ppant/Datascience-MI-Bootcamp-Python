
# coding: utf-8
# Solutions: Code for SF Salary exercises using nymPy and Pandas
# Pradeep K. Pant @2017
# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../../Pierian_Data_Logo.png' /></a>
# ___

# # SF Salaries Exercise 
# 
# Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

# ** Import pandas as pd.**

# In[2]:


import pandas as pd


# ** Read Salaries.csv as a dataframe called sal.**

# In[3]:

sal = pd.read_csv("Salaries.csv")


# ** Check the head of the DataFrame. **

# In[5]:

sal.head()


# ** Use the .info() method to find out how many entries there are.**

# In[6]:

sal.info


# **What is the average BasePay ?**

# In[11]:

sal['BasePay'].mean()


# In[ ]:




# ** What is the highest amount of OvertimePay in the dataset ? **

# In[12]:

sal['OvertimePay'].max()


# In[11]:




# In[ ]:




# In[ ]:




# ** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **

# In[13]:

sal[sal['EmployeeName']== 'JOSEPH DRISCOLL']


# In[ ]:




# ** How much does JOSEPH DRISCOLL make (including benefits)? **

# In[16]:

sal[sal['EmployeeName']== 'JOSEPH DRISCOLL']['TotalPayBenefits']


# In[ ]:




# ** What is the name of highest paid person (including benefits)?**

# In[27]:

sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']


# In[ ]:




# In[ ]:




# In[ ]:




# ** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**

# In[ ]:




# In[28]:

sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['EmployeeName']


# In[31]:

sal.iloc[sal['TotalPayBenefits'].argmin()]['EmployeeName']


# In[15]:




# In[ ]:




# In[ ]:




# ** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **

# In[34]:

sal.groupby('Year').mean()['BasePay']


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ** How many unique job titles are there? **

# In[54]:

sal['JobTitle'].nunique()


# ** What are the top 5 most common jobs? **

# In[53]:

sal['JobTitle'].value_counts().head(5)


# ** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **

# In[52]:

sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)


# ** How many people have the word Chief in their job title? (This is pretty tricky) **

# In[20]:

def checkStringForWord (title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False


# In[21]:

sum(sal['JobTitle'].apply(lambda x:checkStringForWord(x)))


# ** Bonus: Is there a correlation between length of the Job Title string and Salary? **

# In[59]:

sal['title_len'] =  sal['JobTitle'].apply(len)


# In[60]:

sal[['JobTitle','title_len']].corr()


# In[23]:




# # Great Job!
