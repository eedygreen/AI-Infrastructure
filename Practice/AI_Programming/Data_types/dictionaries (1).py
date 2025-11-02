#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1: Count Model Types
# 
# In this exercise, you'll count the number of specific model types in a dataset. Use the dictionary of models and the list of model types to count the total number of each model type, but do not count other items in the dataset.
# 
# **Example Input**:
# ```python
# model_counts = {'logistic_regression': 4, 'decision_tree': 19, 'random_forest': 3, 'datasets': 8}
# model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
# ```
# 
# **Example Output**:
# ```python
# result = 26
# ```

# In[3]:


result = 0
model_counts = {'logistic_regression': 4, 'decision_tree': 19, 'random_forest': 3, 'datasets': 8}
model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']

# Iterate through the dictionary
# TODO
for model, count in model_counts.items():
    if model in model_types:
        result += count

### Notebook grading
def get_solution(model_counts, model_types):
    result = 0
    for model, count in model_counts.items():
        if model in model_types:
            result += count
    return result

correct_answer = get_solution(model_counts, model_types)
if result == correct_answer:
    print("Nice work!")
else:
    print("Try again. That doesn't look like what expected.")


# ## Exercise 2: Validate Model Type Counting Function
# 
# In this exercise, you'll validate that your solution works with any dictionary of items to count the number of specific model types in the dataset.
# 
# **Example Input 1**:
# ```python
# model_counts = {'support_vector_machine': 5, 'neural_network': 19, 'random_forest': 3, 'datasets': 8, 'linear_regression': 4}
# model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
# ```
# 
# **Example Input 2**:
# ```python
# model_counts = {'naive_bayes': 5, 'k_means': 2, 'random_forest': 3, 'datasets': 8, 'decision_tree': 4}
# model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
# ```
# 
# **Example Input 3**:
# ```python
# model_counts = {'k_means': 2, 'datasets': 3, 'support_vector_machine': 8, 'logistic_regression': 4, 'pandas': 10}
# model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
# ```

# In[12]:


# Example 1
result = 0
dataset_items = {'support_vector_machine': 5, 'neural_network': 19, 'random_forest': 3, 'datasets': 8, 'linear_regression': 4}
model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']

# Your previous solution here
# TODO

for key, value in dataset_items.items():
    if key in model_types:
        result += value

print(result)

# Example 2
result = 0
dataset_items = {'naive_bayes': 5, 'k_means': 2, 'random_forest': 3, 'datasets': 8, 'decision_tree': 4}
model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']

# Your previous solution here
# TODO
for model, count in dataset_items.items():
    if model in model_types:
        result += count
print(result)

# Example 3
result = 0
dataset_items = {'k_means': 2, 'datasets': 3, 'support_vector_machine': 8, 'logistic_regression': 4, 'pandas': 10}
model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']

# Your previous solution here
# TODO
for model, count in dataset_items.items():
    if model in model_types:
        result += count
print(result)


# ## Exercise 3: Count Model Types and Other Items
# 
# In this exercise, you'll count both the number of specific model types and other items in the dataset.
# 
# **Example Input**:
# ```python
# model_counts = {'logistic_regression': 4, 'decision_tree': 19, 'random_forest': 3, 'datasets': 8}
# model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
# ```
# 
# **Example Output**:
# ```python
# model_count = 26
# non_model_count = 8
# ```

# In[17]:


model_count, non_model_count = 0, 0
model_counts = {'logistic_regression': 4, 'decision_tree': 19, 'random_forest': 3, 'datasets': 8}
model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']

# Iterate through the dictionary
# TODO
for model, value in model_counts.items():
    if model in model_types:
        model_count += value
    else:
        non_model_count += value

### Notebook grading
def get_solution(model_counts, model_types):
    model_count, non_model_count = 0, 0
    for model, count in model_counts.items():
        if model in model_types:
            model_count += count
        else:
            non_model_count += count
    return model_count, non_model_count

correct_model, correct_non_model = get_solution(model_counts, model_types)
if model_count == correct_model and non_model_count == correct_non_model:
    print("Nice work!")
else:
    print("Try again. That doesn't look like what expected.")


# ## Exercise 4: Calculate Total Model Parameters
# 
# In this exercise, you'll write a loop that iterates over a dictionary of models and their parameters, calculating the total number of parameters used by models of specific types.
# 
# **Example Input**:
# ```python
# model_parameters = {'logistic_regression': 100, 'decision_tree': 200, 'random_forest': 300, 'datasets': 50}
# model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
# ```
# 
# **Example Output**:
# ```python
# total_parameters = 600
# ```

# In[19]:


model_parameters = {'logistic_regression': 100, 'decision_tree': 200, 'random_forest': 300, 'datasets': 50}
model_types = ['logistic_regression', 'decision_tree', 'random_forest', 'support_vector_machine']
total_parameters = 0

# Iterate through the dictionary
# TODO
for model, value in model_parameters.items():
    if model in model_types:
        total_parameters += value

### Notebook grading
def get_solution(model_parameters, model_types):
    total_parameters = 0
    for model, params in model_parameters.items():
        if model in model_types:
            total_parameters += params
    return total_parameters

correct_total_parameters = get_solution(model_parameters, model_types)
if total_parameters == correct_total_parameters:
    print("Nice work!")
else:
    print("Try again. That doesn't look like what expected.")


# ## Exercise 5: Separate Models by Type
# 
# In this exercise, you'll write a loop that iterates over a dictionary of models and separates them into different categories based on their types.
# 
# **Example Input**:
# ```python
# model_info = {'model_a': 'regression', 'model_b': 'classification', 'model_c': 'clustering', 'model_d': 'regression'}
# model_categories = {'regression': 0, 'classification': 0, 'clustering': 0}
# ```
# 
# **Example Output**:
# ```python
# model_categories = {'regression': 2, 'classification': 1, 'clustering': 1}
# ```

# In[22]:


model_info = {'model_a': 'regression', 'model_b': 'classification', 'model_c': 'clustering', 'model_d': 'regression'}
model_categories = {'regression': 0, 'classification': 0, 'clustering': 0}

# Iterate through the dictionary
# TODO
for model, category in model_info.items():
    if category in model_categories:
        model_categories[category] += 1

### Notebook grading
def get_solution(model_info, model_categories):
    model_categories = {'regression': 0, 'classification': 0, 'clustering': 0}
    for model, category in model_info.items():
        if category in model_categories:
            model_categories[category] += 1
    return model_categories

correct_model_categories = get_solution(model_info, model_categories)
if model_categories == correct_model_categories:
    print("Nice work!")
else:
    print("Try again. That doesn't look like what expected.")


# ## Exercise 6: Filter Models by Accuracy
# 
# In this exercise, you'll write a loop that iterates over a dictionary of models and their accuracies, filtering out models that do not meet a specified accuracy threshold.
# 
# **Example Input**:
# ```python
# model_accuracies = {'model_a': 0.95, 'model_b': 0.80, 'model_c': 0.85, 'model_d': 0.90}
# accuracy_threshold = 0.85
# ```
# 
# **Example Output**:
# ```python
# filtered_models = {'model_a': 0.95, 'model_c': 0.85, 'model_d': 0.90}
# count = 3
# ```

# In[27]:


model_accuracies = {'model_a': 0.95, 'model_b': 0.80, 'model_c': 0.85, 'model_d': 0.90}
accuracy_threshold = 0.85
filtered_models = {}
count = 0

# Iterate through the dictionary
# TODO

for model, threshold in model_accuracies.items():
    if threshold >= accuracy_threshold:
        filtered_models[model] = threshold
        count += 1

### Notebook grading
def get_solution(model_accuracies, accuracy_threshold):
    filtered_models = {}
    count = 0
    for model, accuracy in model_accuracies.items():
        if accuracy >= accuracy_threshold:
            filtered_models[model] = accuracy
            count += 1
    return filtered_models, count

correct_filtered_models, correct_count = get_solution(model_accuracies, accuracy_threshold)
if filtered_models == correct_filtered_models and count == correct_count:
    print("Nice work!")
else:
    print("Try again. That doesn't look like what expected.")


# In[ ]:




