#!/usr/bin/env python
# coding: utf-8

# ## Quiz 1: Evaluating Model Performance
# ### Scenario:
# You are working on a project where you need to analyze the average prediction accuracy of a machine learning model over the last three months. The accuracy scores for the last three months are 85.5%, 90.2%, and 87.3%.
# 
# ### Task:
# Write an expression to calculate the mean prediction accuracy over these three months.

# In[10]:


# Accuracy scores for the last three months
accuracy_scores = [85.5, 90.2, 87.3]

# Calculate the average accuracy
average_accuracy = (accuracy_scores[0] + accuracy_scores[1] + accuracy_scores[2])/len(accuracy_scores) #replace `None` with your code

print(f"The average prediction accuracy over the last three months is {average_accuracy:.2f}%")


# ## Quiz 2: Managing Data Storage for Machine Learning Projects
# ### Scenario:
# You are optimizing the data storage for a machine learning project. You have two datasets of image files for training. One dataset contains 4000 images, and the other contains 3000 images. Each image is 256KB in size. Additionally, you receive a new storage device with 2048MB capacity.
# 
# ### Tasks:
# 
# 1. Calculate the total storage needed for both datasets in MB.
# 2. Determine how much storage will be left after storing all images on the new device in MB.

# In[13]:


# Sizes of datasets (in images)
dataset1_size = 4000
dataset2_size = 3000

# Size of each image in KB
image_size_kb = 256

# Calculate total storage needed (in KB)
total_storage_kb = (dataset1_size + dataset2_size) * image_size_kb # None #replace `None` with your code

# Convert total storage needed to MB
total_storage_mb = total_storage_kb / 1024
print(f"Total storage needed: {total_storage_mb:.2f} MB")

# Storage device capacity (in MB)
device_capacity_mb = 2048

# Calculate leftover storage in MB
leftover_storage_mb =  total_storage_mb - device_capacity_mb #replace `None` with your code
print(f"Storage left after storing all images: {leftover_storage_mb:.2f} MB")


# In[ ]:




