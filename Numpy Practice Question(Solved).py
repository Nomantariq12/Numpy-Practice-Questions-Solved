#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


# Task no 1
# create 2d array from 1,12 range 
# dimension should be 6row 2 columns  
# and assign this array values in x values in x variable
# Hint: you can use arange and reshape numpy methods
def function1():
    x = np.arange(1, 13).reshape(6, 2)
    return x
function1()

"""
    expected output:
    [[ 1  2]
    [ 3  4]
    [ 5  6]
    [ 7  8]
    [ 9 10]
    [11 12]]
    """


# In[4]:


np.arange(10, 37)


# In[106]:


# Task 2
#create 3D array (3,3,3)
#must data type should have float64
#array value should be satart from 10 and end with 36 (both included)
# Hint: dtype, reshape

def function2():
    x = np.arange(10, 37, dtype="float64").reshape(3, 3, 3)
    return x
function2()

"""
Expected: out put
array([[[10., 11., 12.],
        [13., 14., 15.],
        [16., 17., 18.]],

       [[19., 20., 21.],
        [22., 23., 24.],
        [25., 26., 27.]],

       [[28., 29., 30.],
        [31., 32., 33.],
        [34., 35., 36.]]])    
"""


# In[50]:


a = np.arange(1, 100*10+1).reshape(100,10)
b = a[(a % 5 == 0) & (a % 7 == 0)]
b


# In[7]:


# Task 3
#extract those numbers from given array. those are must exist in 5,7 Table
#example [35,70,105,..]

def function3():
    a = np.arange(1, 100*10+1).reshape(100, 10)
    b = a[a%5==0]
    x = b[b%7==0]
    return x
function3()

"""
    Expected Output:
     [35,  70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455,
       490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 840, 875, 910,
       945, 980] 
    """ 


# In[8]:


x = np.arange(9).reshape(3,3)
x[:, [0,1]] = x[:, [1,0]]
x


# In[9]:


# Task 4
#Swap columns 1 and 2 in the array arr.

def function4():
    x = np.arange(9).reshape(3,3)
    x[:,[0,1]] = x[:, [1,0]]
    return x
function4()

"""
    Expected Output:
          array([[1, 0, 2],
                [4, 3, 5],
                [7, 6, 8]])
    """ 


# In[10]:


get_ipython().run_line_magic('pinfo2', 'np.swapaxes')


# In[11]:


x = np.arange(10, 37).reshape(3, 3, 3)
x


# In[12]:


x[:, :, [0,1]] = x[:, :, [1,0]]


# In[13]:


x


# In[14]:


x = np.arange(3*3*2*2).reshape(3,3,2,2)
x


# In[15]:


x[:,:,:,[0, 1]] = x[:,:,:,[1,0]]


# In[16]:


x


# In[17]:


# Task 5
#Create a null vector of size 20 with 4 rows and 5 columns with numpy function

def function5():
    z = np.zeros((5, 4), dtype="int32")
    return z
function5()

"""
    Expected Output:
          array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])
    """ 


# In[18]:


# Task 6
# Create a null vector of size 10 but the fifth and eighth 
#value which is 10,20 respectively

def function6():
    x = np.zeros(10, dtype="int32")
    x[[4, 7]] = 10, 20
    return x
function6()


# In[19]:


# Task 7
#  Create an array of zeros with the same shape and type as X. Dont use reshape method

def function7():
    x = np.arange(4, dtype=np.int64)
    return np.zeros_like(x)
function7()

"""
    Expected Output:
          array([0, 0, 0, 0], dtype=int64)
    """ 


# In[20]:


# Task 8
# Create a new array of 2x5 uints, filled with 6.

def function8():
    x = np.full((2, 5), fill_value=6, dtype=np.uint64)
    return x
function8()

"""
     Expected Output:
              array([[6, 6, 6, 6, 6],
                     [6, 6, 6, 6, 6]], dtype=uint32)
     """ 


# In[21]:


# Task 9

def function9():
    x = np.arange(2, 101, 2)
    return x
function9()

"""
     Expected Output:
              array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
                    28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,
                    54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,
                    80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])
     """ 


# In[22]:


def pra_function9():
    x = np.arange(2, 101)
    a = x[x % 2 == 0]
    return a
pra_function9()

"""
     Expected Output:
              array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
                    28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,
                    54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,
                    80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])
     """ 
    


# In[25]:


arr = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]])
brr = np.array([1, 2, 3])
print(arr)
print()
print(brr)


# In[27]:


arr.T - brr


# In[28]:


# Task 10
# Subtract the 1d array brr from the 2d array arr, 
#such that each item of brr subtracts from respective row of arr.

def function10():
    arr = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]])
    brr = np.array([1, 2, 3])
    subt = arr.T - brr
    return subt
function10()

 """
     Expected Output:
               array([[2 2 2]
                      [2 2 2]
                      [2 2 2]])
     """ 


# In[33]:


# Task11
# Replace all odd numbers in arr with -1 without changing arr.

def function11():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arr[arr%2 != 0] = -1
    ans = arr
    return ans
function11()

"""
     Expected Output:
              array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
     """ 


# In[43]:


# Task 12
# Create the following pattern without hardcoding. Use only numpy 
#functions and the below input array arr.
# HINT: use stacking concept

def function12():
    arr = np.array([1, 2, 3])
    ans = np.repeat(arr, 3)
    ans1 = np.tile(arr, 3)
    ans3 = np.hstack((ans, ans1))
    return ans3
function12()

 """
     Expected Output:
              array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
     """ 


# In[48]:


# Task 13
# Set a condition which gets all items between 5 and 10 from arr.

def function13():
    arr = np.array([2, 6, 1, 9, 10, 3, 27])
    ans = arr[(arr > 5) & (arr < 10)]
    return ans
function13()

"""
     Expected Output:
              array([6, 9])
     """ 


# In[53]:


# Task 14
# Create an 8X3 integer array from a range between 10 to 34 
#such that the difference between each element is 1 and
#then Split the array into four equal-sized sub-arrays.
# Hint use split method

def function14():
    arr = np.arange(10, 34, 1)
    ans = np.split(arr, 4)
    return ans
function14()

"""
     Expected Output:
       [array([[10, 11, 12],[13, 14, 15]]), 
        array([[16, 17, 18],[19, 20, 21]]), 
        array([[22, 23, 24],[25, 26, 27]]), 
        array([[28, 29, 30],[31, 32, 33]])]
     """ 


# In[69]:


arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])
print(arr)
arr[np.argsort(arr[:, 1])]


# In[72]:


# Task 15
#Sort following NumPy array by the second column

def function15():
    arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])
    ans = arr[np.argsort(arr[:, 1])]
    return ans
function15()

"""
     Expected Output:
           array([[-4,  1,  7],
                   [ 8,  2, -2],
                   [ 6,  3,  9]])
     """ 


# In[75]:


x = np.array([[1], [2], [3]])
y = np.array([[2], [3], [4]])

np.dstack((x, y))


# In[81]:


# Task 16
#Write a NumPy program to join a sequence of arrays along depth.

def function16():
    x = np.array([[1], [2], [3]])
    y = np.array([[2], [3], [4]])
    ans = np.dstack((x, y))
    return ans
function16()

"""
     Expected Output:
                [[[1 2]]

                 [[2 3]]

                 [[3 4]]]
     """ 


# In[82]:


# task 17
# replace numbers with "YES" if it divided by 3 and 5
# otherwise it will be replaced with "NO"
# Hint: np.where

def function17():
    arr = np.arange(1, 10*10+1).reshape(10, 10)
    return np.where((arr % 3 == 0)&(arr % 5 == 0), "Yes", "No")
function17()

#Excpected Out
"""
array([['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']],
      dtype='<U3')
"""


# In[83]:


arr = np.arange(1, 10*10+1).reshape(10, 10)
np.where((arr % 3 == 0)&(arr % 5 == 0), arr, "No")


# In[94]:


students = np.arange(100)
piaic = np.array([5, 20, 50, 200, 488, 4833, 5000])
len(np.intersect1d(students, piaic))


# In[95]:


# Task 18
# count values of "students" are exist in "piaic"

def function18():
    students = np.arange(100)
    piaic = np.array([5, 20, 50, 200, 488, 4833, 5000])
    x = len(np.intersect1d(students, piaic))
    return x
function18()

#Expected output: 3


# In[97]:


x = np.arange(1, 26).reshape(5, 5)
w = np.copy(x).T
w


# In[98]:


# Task 19
#Create variable "X" from 1,25 (both are included) range values
#Convert "X" variable dimension into 5 rows and 5 columns
#Create one more variable "W" copy of "X" 
#Swap "W" row and column axis (like transpose)
# then create variable "b" with value equal to 5
# Now return output as "(X*W)+b:

def function19():
    x = np.arange(1, 26).reshape(5, 5)
    w = np.copy(x).T
    b = 5
    return (x*w)+b
function19()

#expected output
    """
    array([[  6,  17,  38,  69, 110],
       [ 17,  54, 101, 158, 225],
       [ 38, 101, 174, 257, 350],
       [ 69, 158, 257, 366, 485],
       [110, 225, 350, 485, 630]])
    """


# In[104]:


# Task 20
#apply fuction "abc" on each value of Array "X"

def function20():
    x = np.arange(1, 11)
    def abc(x):
        return x*2+3-2
    return abc(x)
    
function20()

#Expected Output: array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21])


# In[ ]:




