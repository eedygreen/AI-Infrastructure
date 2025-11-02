#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1: Tokenizing a Sentence
# 
# **Problem Statement**:
# Given a sentence, tokenize it into individual words and print each word on a new line. This is a common step in NLP to prepare text data for further analysis.
# 
# **Instructions**:
# 1. Create a list of words from a sentence.
# 2. Use a for loop to print each word on a new line.
# 
# **Example Input**:
# ```python
# sentence = "the quick brown fox jumped over the lazy dog"
# ```

# In[1]:


# Define the sentence
sentence = "the quick brown fox jumped over the lazy dog"

# Tokenize the sentence into words
words = sentence.split()

# Print each word on a new line
# TODO
for word in words:
    print(word)


# ## Exercise 2: Batching Data for Model Training
# 
# In this exercise, you will simulate the process of batching data for model training, where data is processed in fixed-size batches.
# 
# **Problem Statement**:
# Given a dataset, divide it into batches of a specified size and print each batch. This simulates the process of batching data during model training.
# 
# **Instructions**:
# 1. Create a list of numbers from 1 to 30.
# 2. Use a for loop to divide the list into batches of 5 items each and print each batch.

# In[5]:


# Define the dataset
data = list(range(1, 31))

# Define the batch size
batch_size = 5

# Process the data in batches
# TODO
for i in range (0, len(data), batch_size):
    dataset = data[i:i + batch_size]
    print(f"Batch {i //batch_size +1}: {dataset}")
        


# In[ ]:




