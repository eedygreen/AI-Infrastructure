#!/usr/bin/env python
# coding: utf-8

# ## Understanding Hyperparameters in Machine Learning
# Hyperparameters are settings or configurations that you set before the training process begins, which control the behavior and performance of machine learning models. Unlike model parameters, which are learned from the data during training, hyperparameters are predefined and must be set manually. They play a crucial role in model performance and can significantly impact the results.
# 
# ### Common Hyperparameters:
# 1. Learning Rate: Controls how much to change the model in response to the estimated error each time the model weights are updated.
# 2. Batch Size: Determines the number of training samples used in one iteration.
# 3. Number of Epochs: Specifies how many times the learning algorithm will work through the entire training dataset.
# 4. Regularization Parameters: Help prevent overfitting by adding a penalty to the loss function.
# 5. Momentum: Accelerates the gradient vectors in the right directions, leading to faster converging.
# 
# ### Importance of Hyperparameters:
# Setting the right hyperparameters is essential for model training because they directly influence the training process and the performance of the model. Proper tuning of hyperparameters can lead to better model accuracy, faster convergence, and overall improved results.

# ## Quiz: Adjusting AI Model Parameters
# ### Scenario:
# You are working with an AI model and need to adjust various parameters and metrics during the training process. Use the assignment operators to modify these values based on the given instructions.
# 
# ### Task:
# Follow the instructions in the comments to modify the variables. After each comment, write a line of code that implements the instruction.

# In[24]:


# The initial learning rate for the model
learning_rate = 0.01
# The initial training loss of the model
training_loss = 0.8

# Use the multiplication assignment operator to decrease 
# the learning_rate variable by 10% to account for better stability
learning_rate *= 0.9 # TODO

### Notebook grading
if learning_rate == 0.01 * 0.9:
    print("Nice job! You can use the multiplication assignment operator like this: learning_rate *= 0.9")
elif learning_rate == 0.01 * 1.1:
    print("Sorry! learning_rate is too high. It looks like you increased the learning rate by 10% instead of decreasing it. Try again")        
else:
    print("Sorry! learning_rate is incorrect. Try again and make sure to use the *= assignment operator")


# In[7]:


# Use the addition assignment operator to add 
# a small value (0.02) to the training_loss variable due to noisy data
training_loss += 0.02 # TODO

### Notebook grading
if training_loss == 0.8 + 0.02:
    print("Nice job! You can use the addition assignment operator like this: training_loss += 0.02")
elif training_loss == 0.8:
    print("Sorry! training_loss is too low. Did you add the noise? Try again")        
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the += assignment operator")


# In[28]:


# Use the multiplication assignment operator to increase 
# training_loss by 5% to account for overfitting
training_loss *= 1.05 # TODO

### Notebook grading
if training_loss == (0.8 + 0.02) * 1.05:
    print("Nice job! You can use the multiplication assignment operator like this: training_loss *= 1.05")
elif training_loss == 0.8 + 0.02:
    print("Sorry! training_loss is too low. Did you account for overfitting? Try again")        
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the *= assignment operator")


# In[29]:


# Use the multiplication assignment operator to decrease 
# training_loss by 5% to account for regularization
training_loss *= 0.95 # TODO

### Notebook grading
if training_loss == (0.8 + 0.02) * 1.05 * 0.95:
    print("Nice job! You can use the multiplication assignment operator like this: training_loss *= 0.95")
elif training_loss == (0.8 + 0.02) * 1.05:
    print("Sorry! training_loss is too high. Did you account for regularization? Try again")        
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the *= assignment operator")


# In[30]:


# Use the subtraction assignment operator to subtract 
# a small value (0.05) from training_loss to account 
# for improved generalization due to more training data.
training_loss -= 0.05 # TODO

### Notebook grading
if training_loss == (0.8 + 0.02) * 1.05 * 0.95 - 0.05:
    print("Nice job! You can use the subtraction assignment operator like this: training_loss -= 0.05")
elif training_loss == (0.8 + 0.02) * 1.05 * 0.95:
    print("Sorry! training_loss is too high. Did you account for the improved generalization? Try again")        
else:
    print("Sorry! training_loss is incorrect. Try again and make sure to use the -= assignment operator")


# ### Understanding Hyperparameters: Learning Rate
# Learning Rate is a crucial hyperparameter in machine learning models that determines the step size at each iteration while moving towards a minimum of the loss function. It is common to use [scientific notation](https://en.wikipedia.org/wiki/Scientific_notation) for learning rates because they are often very small values. For example, a learning rate of `0.001` can be represented as `1𝑒−3`.

# In[21]:


# Initial learning rate
learning_rate = 1e-3  # Equivalent to 0.001

# Update the learning rate by decreasing it by 10%
learning_rate = 1e-2 # TODO

print(f"Updated learning rate: {learning_rate}")


# In[ ]:




