#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1: Create Model Identifiers
# 
# In this exercise, you'll write a loop that iterates over a list of model names to create a list of model identifiers. To create an identifier for each model, make everything lowercase and replace spaces with underscores.
# 
# ### Problem Statement
# 
# Given a list of model names, create a list of model identifiers by converting each name to lowercase and replacing spaces with underscores.
# 
# **Example Input**:
# ```python
# model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]
# ```
# 
# **Example Output**:
# ```python
# ["logistic_regression", "decision_tree", "random_forest", "support_vector_machine"]
# ```

# In[3]:


model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]
model_identifiers = []
# Write your for loop here
# TODO
for name in model_names:
    model_identifiers.append(name.lower().replace(" ", "_"))
print(model_identifiers)

### Notebook grading
if model_identifiers == ["logistic_regression", "decision_tree", "random_forest", "support_vector_machine"]:
    print("Nice work!")
else:
    print("Not quite! Did you append each new identifier to the list `model_identifiers`?")


# ## Exercise 2: Modify Model Identifiers with Range
# 
# In this exercise, you'll write a loop that uses `range()` to iterate over the positions in `model_identifiers` to modify the list. Change each model name to be lowercase and replace spaces with underscores.
# 
# ### Problem Statement
# 
# Given a list of model names, modify the list so that each model name is converted to lowercase and spaces are replaced with underscores.
# 
# **Example Input**:
# ```python
# model_identifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]
# ```
# 
# **Example Output**:
# ```python
# ["logistic_regression", "decision_tree", "random_forest", "support_vector_machine"]
# ```

# In[14]:


model_identifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]

# Write your for loop here
# TODO
for model_identifier in range(len(model_identifiers)):
    model_identifiers[model_identifier] = model_identifiers[model_identifier].lower().replace(" ", "_")
### Notebook grading
if model_identifiers == ["logistic_regression", "decision_tree", "random_forest", "support_vector_machine"]:
    print("Nice work!")
else:
    print("Not quite! Did you modify each element in the list `model_identifiers`?")


# ## Exercise 3: Prediction Tag Counter
# 
# In this exercise, you'll write a loop that iterates over a list of strings, `predictions`, and counts how many of them are in the format of predictions (strings that start with "Predicted:").
# 
# ### Problem Statement
# 
# Given a list of strings, count how many of them start with "Predicted:".
# 
# **Example Input**:
# ```python
# predictions = ['Predicted: 0.95', 'Actual: 0.90', 'Predicted: 0.85']
# ```
# 
# **Example Output**:
# ```python
# count = 2
# ```

# In[13]:


predictions = ['Predicted: 0.95', 'Actual: 0.90', 'Predicted: 0.85']
count = 0

# Write your for loop here
# TODO
for prediction in range(len(predictions)):
    if predictions[prediction].startswith("Predicted"):
        count += 1
        
### Notebook grading
if count == 2:
    print("Nice work!")
else:
    print("Not quite! Did you track the number of predictions with `count`?")


# ## Exercise 4: Create an HTML List of Model Metrics
# 
# In this exercise, you'll write a loop that iterates over a list of model metrics and creates a single string, `html_str`, which is an HTML list.
# 
# ### Problem Statement
# 
# Given a list of model metrics, create a single string that represents an HTML list.
# 
# **Example Input**:
# ```python
# metrics = ['Accuracy: 0.95', 'Precision: 0.92', 'Recall: 0.88']
# ```
# 
# **Example Output**:
# ```
# <ul>
# <li>Accuracy: 0.95</li>
# <li>Precision: 0.92</li>
# <li>Recall: 0.88</li>
# </ul>
# ```
# 
# 

# In[19]:


metrics = ['Accuracy: 0.95', 'Precision: 0.92', 'Recall: 0.88']
html_str = "<ul>\n"

# Write your code here
# TODO
for metric in metrics:
    html_str += "<li>{}</li>\n".format(metric)
    html_str += "</ul>"

# Print the resulting HTML string
print(html_str)


# In[ ]:




