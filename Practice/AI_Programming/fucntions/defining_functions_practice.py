#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1: Calculating Population Density of Data Centers
# #### Scenario:
# You are responsible for managing the storage capacities of data centers in various regions. To optimize resource allocation, you need to calculate the data storage density for each region. The storage density is defined as the amount of data (in TB) stored per square kilometer of land area.
# 
# #### Task:
# Write a function named `data_storage_density` that takes two arguments, `data_amount` (in TB) and `land_area` (in square kilometers), and returns the data storage density calculated from those values.

# In[2]:


# Write your function here
def data_storage_density(data_amount, land_area):
    
    return data_amount/land_area

### Notebook grading
import types
explanation_str = '''Your function produced the wrong result when called like this: {}\t
The expected output is: {}'''
if 'data_storage_density' not in locals():
    print("Your code doesn't define `data_storage_density`. Check your spelling.")
elif not isinstance(data_storage_density, types.FunctionType):
    print("`data_storage_density` should be a function.")
elif data_storage_density(10, 1) != 10 / 1:
    print(explanation_str.format('data_storage_density(10, 1)', 10. / 1))
elif data_storage_density(864816, 121.4) != 864816 / 121.4:
    print(explanation_str.format('data_storage_density(864816, 121.4)', 864816. / 121.4))
elif data_storage_density(1234321, 42) != 1234321 / 42:
    print(explanation_str.format('data_storage_density(1234321, 42)', 1234321. / 42))
else:
    print("Nicely done! You can view my solution on the next page.")


# ### Exercise 2: Readable Time for Data Processing
# #### Scenario:
# You are optimizing a machine learning pipeline, and you need to report the processing time of a dataset in a human-readable format. The time is provided in days, and you need to convert it to weeks and days to better communicate the processing schedule to the stakeholders.
# 
# #### Task:
# Write a function named `readable_processing_time`. The function should take one argument, an integer `days`, and return a string that says how many weeks and days that is.

# In[5]:


# Write your function here
def readable_processing_time(days):
    """Print the number of weeks and days in a number of days"""
    weeks = days // 7
    reminder = days % 7
    return f"{weeks} week(s) and {reminder} day(s)."

### Notebook grading
import random
def readable_processing_time_soltn(days):
    """Print the number of weeks and days in a number of days."""
    weeks = days // 7
    remainder = days % 7
    return "{} week(s) and {} day(s).".format(weeks, remainder)

explanation_str = '''Your function produced the wrong result when called like this: readable_processing_time({}). The expected output is: {}'''

test_cases = [1, 6, 7, 9, random.randint(100, 10000)]
test_failed = False

if 'readable_processing_time' not in locals():
    test_failed = True
    print("Your code doesn't define `readable_processing_time`. Check your spelling.")
elif not isinstance(readable_processing_time, types.FunctionType):
    test_failed = True
    print("`readable_processing_time` should be a function.")
else:
    for case in test_cases:
        if readable_processing_time(case) != readable_processing_time_soltn(case):
            test_failed = True
            print(explanation_str.format(case, readable_processing_time_soltn(case)))
            break

    if not test_failed:
        print("Nicely done! You can view my solution on the next page.")


# In[ ]:




