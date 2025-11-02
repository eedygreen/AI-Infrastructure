#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# 
# You are training a machine learning model, and you want to stop the training process if the validation loss does not improve for a certain number of epochs. You have a list of tuples where each tuple represents an epoch's performance, with the first element being the epoch number and the second element being the validation loss.
# 
# **List of Epoch Performances:**
# ```python
# epoch_performances = [(1, 0.5), (2, 0.45), (3, 0.47), (4, 0.44), (5, 0.43), (6, 0.42), (7, 0.44), (8, 0.41), (9, 0.42), (10, 0.43)]
# ```
# 
# You need to implement two methods to stop the training:
# 
# 1. **Method 1:** Stop the training immediately when the validation loss increases compared to the previous epoch.
# 2. **Method 2:** Stop the training when the validation loss has not improved for a certain number of epochs (`patience`).
# 
# ### Instructions
# 
# 1. **Method 1: Immediate Stop**
# 
#    Write a loop that iterates through the `epoch_performances` list and stops immediately when the validation loss increases compared to the previous epoch.
# 
# 
# 2. **Method 2: Patience**
# 
#    Write a loop that iterates through the `epoch_performances` list and stops when the validation loss has not improved for a specified number of epochs (`patience`).
# 
# 
# 3. **Output:**
# 
#    Both methods should output the number of epochs run and the reason for stopping.
# 
# ### Explanation
# 
# - **Early Stopping:** A technique used to halt training when the model starts to overfit the training data.
# - **Validation Loss:** A measure of how well the model performs on a validation dataset.
# - **Patience:** The number of epochs to wait for an improvement in validation loss before stopping the training.
# 
# These methods help to understand the importance of monitoring model performance and stopping the training process to avoid overfitting.
# 

# In[2]:


epoch_performances = [(1, 0.5), (2, 0.45), (3, 0.47), (4, 0.44), (5, 0.43), (6, 0.42), (7, 0.44), (8, 0.41), (9, 0.42), (10, 0.43)]


# In[3]:


print("METHOD 1: Immediate Stop")
best_loss = float('inf')
epochs_ran = 0

for epoch, val_loss in epoch_performances:
    print(f"Epoch {epoch} - Validation Loss: {val_loss}")
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_ran += 1
    else:
        print(f"Stopping early at epoch {epoch} as validation loss increased.")
        break

print(f"\nTotal epochs ran: {epochs_ran}")


# In[ ]:


print("\nMETHOD 2: Patience")
patience = 2
best_loss = float('inf')
epochs_ran = 0
patience_counter = 0

for epoch, val_loss in epoch_performances:
    print(f"Epoch {epoch} - Validation Loss: {val_loss}")
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        epochs_ran += 1
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch} as validation loss did not improve for {patience} epochs.")
            break
        epochs_ran += 1

print(f"\nTotal epochs ran: {epochs_ran}")

