
# coding: utf-8

# # Python Crash Course Exercises 
# Solutions: # Pradeep K. Pant @2017
# 
# This is an optional exercise to test your understanding of Python Basics. If you find this extremely challenging, then you probably are not ready for the rest of this course yet and don't have enough programming experience to continue. I would suggest you take another course more geared towards complete beginners, such as [Complete Python Bootcamp](https://www.udemy.com/complete-python-bootcamp)

# ## Exercises
# 
# Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

# ** What is 7 to the power of 4?**

# In[2]:

print (7**4)


# In[1]:




# In[ ]:




# In[ ]:




# In[ ]:




# ** Split this string:**
# 
#     s = "Hi there Sam!"
#     
# **into a list. **

# In[5]:

s = "Hi there Sam!"


# In[11]:

my_list = s.split()


# In[12]:

my_list[2] = 'dad!'


# In[ ]:




# In[3]:




# ** Given the variables:**
# 
#     planet = "Earth"
#     diameter = 12742
# 
# ** Use .format() to print the following string: **
# 
#     The diameter of Earth is 12742 kilometers.

# In[14]:

planet = "Earth"
diameter = 12742


# In[16]:

print('The diameter of {pla} is {dia} kilometers.'.format(pla=planet,dia=diameter))


# In[17]:

print('The diameter of {pla} is {dia} kilometers.'.format(pla=planet,dia=diameter))


# ** Given this nested list, use indexing to grab the word "hello" **

# In[18]:

lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]


# In[23]:

lst[3][1][2][0]


# In[24]:

lst[3][1][2][0]


# ** Given this nested dictionary grab the word "hello". Be prepared, this will be annoying/tricky **

# In[16]:

d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}


# In[30]:


d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}


# In[40]:

d['k1'][3]['tricky'][3]['target'][3]


# In[33]:




# In[ ]:




# ** What is the main difference between a tuple and a list? **

# In[23]:

# Tuple is immutable


# ** Create a function that grabs the email website domain from a string in the form: **
# 
#     user@domain.com
#     
# **So for example, passing "user@domain.com" would return: domain.com**

# In[78]:

def domainGet(strDomain): 
   return strDomain.split('@')[1]


# In[79]:

out = domainGet('user@domain.com')


# In[80]:

print(out)
    


# In[63]:

domainGet('user@domain.com')


# ** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **

# In[200]:

def findDog(st):
    print(st.lower())
    return 'dog' in st.split() 


# In[201]:

findDog('Is there a dog here?')


# In[ ]:




# ** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **

# In[198]:

def countDog(st):
    st.lower().split()
    return st.count('dog')


# In[199]:

countDog('This dog runs faster than the other dog dude!')


# In[31]:




# In[ ]:




# ** Use lambda expressions and the filter() function to filter out words from a list that don't start with the letter 's'. For example:**
# 
#     seq = ['soup','dog','salad','cat','great']
# 
# **should be filtered down to:**
# 
#     ['soup','salad']

# In[196]:

seq = ['soup','dog','salad','cat','great']


# In[197]:

list(filter(lambda item:item[0]=='s',seq))


# In[35]:




# In[ ]:




# ### Final Problem
# **You are driving a little too fast, and a police officer stops you. Write a function
#   to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
#   If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
#   and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
#   cases. **

# In[193]:

def caught_speeding(speed, is_birthday):
    speed_org = speed
    if is_birthday == True:
        speed = speed-5
    else:
        speed = speed_org        
    if speed <= 60:
        print("No Ticket")
    elif speed>60 and speed<81:
        print ("Small Ticket")
    elif speed>=81:
        print ("Big Ticket")
    print(speed)
pass


# In[195]:

caught_speeding(81,False)


# In[ ]:




# In[42]:

caught_speeding(81,True)


# In[43]:

caught_speeding(81,False)


# # Great job!
