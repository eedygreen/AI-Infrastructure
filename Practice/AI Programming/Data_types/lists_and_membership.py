#!/usr/bin/env python
# coding: utf-8

# ## Quiz: Indexing Image Dimensions
# 
# In this exercise, you'll use list indexing to determine the height of an image based on the integer variable `image_index`, and store that value in the integer variable `image_height`. For example, if image_index is 2, image_height should be set to the height of the third image.
# 
# The dimensions of each image (width, height) are stored in the list `image_dimensions`.

# In[1]:


# Dimensions of each image in the format (width, height)
image_dimensions = [(640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080)]

# The index of the image for which we want to find the height
image_index = 2

# TODO: replace None with appropriate code
# Use list indexing to determine the height for `image_index`
image_height = image_dimensions[image_index][1]

### Notebook grading
correct_answer = 768
if image_height != correct_answer:
    print("Not quite! Did you account for zero-based indexing?")
else:
    print("Nice work! I found image_height like this: `image_dimensions[image_index][1]`.")


# ## Quiz: Slicing Image Filenames
# 
# Select the filenames of the last three images from the list using list slicing notation. 
# 
# Hint: negative indexes work in slices!

# In[6]:


# Filenames of images
image_filenames = ['img_001.jpg', 'img_002.jpg', 'img_003.jpg', 'img_004.jpg', 'img_005.jpg', 'img_006.jpg']

# TODO: Replace None with appropriate code
# Modify this code so it prints the filenames of the last three images
last_three_filenames = image_filenames[-3:]

### Notebook grading
correct_answer = ['img_004.jpg', 'img_005.jpg', 'img_006.jpg']
if last_three_filenames != correct_answer:
    print("Double check your slicing! Your code should print a list of the last three filenames from the original list, and nothing else.")
else:
    print('''My solution is this: `image_filenames[-3:]`
This slice uses a negative index to begin slicing three elements from the end of the list. The end index can be omitted because this slice continues until the end of the list.''')


# ## Quiz: Palindrome Check on Image Classifications
# 
# In this exercise, you'll use list slicing to determine if a sequence of image classifications forms a palindrome. A palindrome is a sequence that reads the same backward as forward.
# 
# The list image_classifications contains the predicted classes of a series of images processed by an image classification model.
# 
# Hint: You need to use list slicing to reverse the list. The slicing syntax `[::-1]` reverses the list.

# In[9]:


# Predicted classes of a series of images
image_classifications = ['dog', 'cat', 'bird', 'cat', 'dog']

# TODO: replace None with appropriate code
# Use list slicing to reverse the list and check if it forms a palindrome
is_palindrome = image_classifications == image_classifications[::-1]

### Notebook grading
correct_answer = True
if is_palindrome != correct_answer:
    print("Not quite! Remember to use list slicing to reverse the list and check if it matches the original list.")
else:
    print("Nice work! I found is_palindrome like this: `image_classifications == image_classifications[::-1]`.")


# In[ ]:




