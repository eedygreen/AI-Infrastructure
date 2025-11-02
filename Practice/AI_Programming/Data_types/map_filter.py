#!/usr/bin/env python
# coding: utf-8

# ### Comparing Execution Time of `map`, `filter`, and Conventional For-Loops
# 
# In AI and data processing, efficiency is crucial. Let's compare the execution times of `map`, `filter`, and conventional for-loops to understand the performance benefits of using higher-order functions.

# In[ ]:


import time

# Sample data
numbers = list(range(1, 1000000))
threshold = 500000

# Conventional for-loop for map equivalent (square of each number)
start_time = time.time()
squares_conventional = []
for num in numbers:
    squares_conventional.append(num ** 2)
end_time = time.time()
conventional_map_time = end_time - start_time

# Using map
start_time = time.time()
squares_map = list(map(lambda x: x ** 2, numbers))
end_time = time.time()
map_time = end_time - start_time

# Conventional for-loop for filter equivalent (numbers greater than threshold)
start_time = time.time()
filtered_conventional = []
for num in numbers:
    if num > threshold:
        filtered_conventional.append(num)
end_time = time.time()
conventional_filter_time = end_time - start_time

# Using filter
start_time = time.time()
filtered_filter = list(filter(lambda x: x > threshold, numbers))
end_time = time.time()
filter_time = end_time - start_time

# Printing the results
print(f"Conventional for-loop (map equivalent) took: {conventional_map_time:.6f} seconds")
print(f"Map function took: {map_time:.6f} seconds")
print(f"Conventional for-loop (filter equivalent) took: {conventional_filter_time:.6f} seconds")
print(f"Filter function took: {filter_time:.6f} seconds")


# #### Understanding Lambda Functions, Map, and Filter
# 
# **Lambda Functions**: Lambda functions are small, anonymous functions defined using the `lambda` keyword. They are often used for short, simple operations that are not reused elsewhere. A lambda function can take any number of arguments but only has one expression.
# 
# **Map Function**: The `map()` function applies a given function to all items in an input list (or any other iterable) and returns an iterator with the results. This is particularly useful for transforming data in a concise and readable manner.
# 
# **Filter Function**: The `filter()` function constructs an iterator from elements of an iterable for which a function returns true. It is commonly used to extract items that meet certain criteria from a dataset.
# 
# Let's explore these concepts through practical exercises related to managing large datasets in AI projects.

# ### Exercise 1: Normalizing Dataset Sizes with Lambda and Map
# 
# #### Scenario:
# 
# You are analyzing datasets of various sizes for an AI project and need to normalize the values in each dataset. **Normalization** ensures that all values are scaled to a similar range, preventing features with larger values from dominating the model. This is a crucial preprocessing step in many AI applications.
# 
# #### Task:
# 
# Normalize the values in each dataset using a **min-max normalization** formula within the `map()` function.
# 
# Normalization Explanation:
# 
# We will apply **min-max normalization**, which scales the values to a range of **0** to **1** using the following formula:
# 
# 
# $$X{\prime} = \frac{X - X_\text{min}}{X_\text{max} - X_\text{min}}$$
# 
# 
# where:
# * $X{\prime}$ is the normalized value,
# * $X$ is the original value,
# * $X_\text{min}$ and $X_\text{max}$	are the minimum and maximum values in the dataset.
# 
# **Example Input:**
# ```python
# numbers = [
#     [34, 63, 88, 71, 29],
#     [90, 78, 51, 27, 45],
#     [63, 37, 85, 46, 22],
#     [51, 22, 34, 11, 18]
# ]
# ```
# **Example Output (After Min-Max Normalization):**
# ```python
# [
#     [0.08, 0.58, 1.00, 0.71, 0.00],
#     [1.00, 0.81, 0.38, 0.00, 0.29],
#     [0.65, 0.24, 1.00, 0.38, 0.00],
#     [1.00, 0.28, 0.57, 0.00, 0.18]
# ]
# ```

# In[ ]:


numbers = [
    [34, 63, 88, 71, 29],
    [90, 78, 51, 27, 45],
    [63, 37, 85, 46, 22],
    [51, 22, 34, 11, 18]
]

# Use a lambda function within map to normalize the datasets
normalized_data = list(map(lambda x: [num / (sum(x) / len(x)) for num in x], numbers))

normalized_data


# ### Transformed Exercise 2: Filtering Datasets by Variance with Lambda and Filter
# #### Scenario:
# You need to filter datasets that have a variance above a specified threshold. Variance is a measure of the dispersion of data points and helps in identifying datasets with significant variability, which can be crucial for certain AI applications.
# 
# #### Task:
# Filter datasets that have a variance above a specified threshold using a lambda function within the `filter()` function.

# In[ ]:


datasets = [
    [34, 63, 88, 71, 29],
    [90, 78, 51, 27, 45],
    [63, 37, 85, 46, 22],
    [51, 22, 34, 11, 18]
]

# Calculate variance for each dataset
def variance(num_list):
    mean_val = sum(num_list) / len(num_list)
    return sum((x - mean_val) ** 2 for x in num_list) / len(num_list)

# Filter datasets with variance above a threshold using a lambda function
threshold = 400
filtered_datasets = list(filter(lambda x: variance(x) > threshold, datasets))

filtered_datasets

