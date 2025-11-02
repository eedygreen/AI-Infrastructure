#!/usr/bin/env python
# coding: utf-8

# ## Exercise: Readable Time for Data Processing
# ### Scenario:
# You are optimizing a machine learning pipeline, and you need to report the processing time of a dataset in a human-readable format. The time is provided in days, and you need to convert it to weeks and days to better communicate the processing schedule to the stakeholders.
# 
# ### Task:
# Write a function named readable_processing_time. The function should take one argument, an integer days, and return a string that says how many weeks and days that is.

# In[4]:


def readable_processing_time(days):
    """ This Function readable_processing_time convert the number of days to human readable format
    INPUT:
        Enter the number of days in figures
        Example, call the function as this readabale_processing_time(43)
    OUTPUT:
        It prints out the humandable out in weeks and days
        Here is the Output from the example call from the INPUT
        6 week(s) and 1 day(s)
    """
    weeks = days // 7
    remainder = days % 7
    return "{} week(s) and {} day(s)".format(weeks, remainder)


### Notebook grading
import inspect
if 'readable_processing_time' not in locals():
    print("Your code doesn't define the `readable_processing_time` function.")
elif inspect.getdoc(readable_processing_time) is None:
    print("Your function doesn't have a docstring! Add one that explains the function's purpose.")
else:
    print("Nicely done! You can view my solution on the next page.")


# In[ ]:




