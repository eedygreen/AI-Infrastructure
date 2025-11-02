#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1: Learning Rate Schedule using While Loops
# 
# #### Problem Statement
# 
# Given an initial learning rate and a decay factor, use a while loop to apply the decay to the learning rate at each epoch until a specified number of epochs is reached.
# 
# **Example Input**:
# ```python
# initial_lr = 0.1
# decay_factor = 0.9
# epochs = 5
# ```
# 
# **Instructions**:
# 1. Initialize the learning rate to `initial_lr`.
# 2. Use a while loop to apply the decay factor to the learning rate for each epoch.
# 3. Print the learning rate for each epoch.

# In[4]:


# Initial learning rate
initial_lr = 0.1
# Decay factor
decay_factor = 0.9
# Number of epochs
epochs = 5

# Initialize current learning rate
current_lr = initial_lr
# Initialize current epoch
current_epoch = 0
while current_epoch < epochs:
    print(f"Epoch {current_epoch + 1}: Learning rate = {current_lr:.5f}")
    current_lr *= decay_factor
    current_epoch += 1

# While loop to apply learning rate decay
while current_epoch < epochs:
    print(f"Epoch {current_epoch + 1}: Learning rate = {current_lr:.5f}")
    # Apply decay to the current learning rate
    current_lr *= decay_factor
    # Increment the current epoch
    current_epoch += 1

# Notebook grading
if abs(current_lr - 0.059049) < 1e-6:
    print("Nice work!")
else:
    print("Not quite. Check your learning rate calculations.")


# ### Exercise 2: Iterating Through Model Parameters using For Loops
# 
# #### Problem Statement
# 
# Given a list of model parameters and their corresponding gradients, use a for loop to update each parameter.
# 
# **Example Input**:
# ```python
# parameters = [0.5, 1.5, -0.5]
# gradients = [0.1, -0.2, 0.05]
# learning_rate = 0.01
# ```
# 
# **Instructions**:
# 1. Use a for loop to iterate through the parameters and gradients.
# 2. Apply a gradient update to each parameter using the learning rate.

# In[13]:


# Model parameters
parameters = [0.5, 1.5, -0.5]
# Corresponding gradients
gradients = [0.1, -0.2, 0.05]
# Learning rate
learning_rate = 0.01

for i in range(len(parameters), len(gradients)):
    parameters[i] -= learning_rate * gradients[i]
    print(f"Updated parameter {i + 1}: {parameters[i]:.5f}")

# For loop to update each parameter
for i in range(len(parameters)):
    parameters[i] -= learning_rate * gradients[i]
    print(f"Updated parameter {i + 1}: {parameters[i]:.5f}")

# Notebook grading
if parameters == [0.499, 1.502, -0.5005]:
    print("Nice work!")
else:
    print("Not quite. Check your parameter updates.")


# In[ ]:




