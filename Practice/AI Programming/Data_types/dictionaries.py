#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Image Classification Models
# 
# In the field of artificial intelligence, particularly in computer vision, image classification is a fundamental task. It involves assigning a label or category to an input image based on its visual content. The success of image classification tasks relies heavily on the architecture of the models used. Among the various types of models, Convolutional Neural Networks (CNNs) have proven to be highly effective. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images through a series of layers.
# 
# In this section, we will discuss several prominent CNN architectures that have significantly advanced the field of image classification. These models include AlexNet, VGG, ResNet, and MobileNet. Each of these models has unique characteristics and innovations that make them suitable for different applications and environments.
# 
# ### Convolutional Neural Networks (CNNs)
# **Convolutional Neural Networks (CNNs)** are a class of deep learning algorithms that are particularly well-suited for image recognition and classification tasks. CNNs automatically and adaptively learn spatial hierarchies of features from input images. They consist of multiple layers, such as convolutional layers, pooling layers, and fully connected layers, which help in detecting edges, textures, shapes, and more complex features. CNNs have been the foundation for many state-of-the-art models in computer vision.
# 
# ### AlexNet
# **AlexNet** is a pioneering convolutional neural network (CNN) architecture that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. Developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, AlexNet consists of eight layers: five convolutional layers followed by three fully connected layers. It introduced the use of ReLU activation functions, dropout layers to prevent overfitting, and data augmentation techniques to improve model generalization.
# 
# ### VGG (Visual Geometry Group)
# **VGG** refers to a series of convolutional neural network models developed by the Visual Geometry Group at the University of Oxford. The VGG architectures, such as VGG-16 and VGG-19, are known for their simplicity and uniform architecture. They use 3x3 convolutional layers stacked on top of each other, with increasing depth. This design allows the network to capture complex features while keeping the number of parameters manageable. VGG models achieved high accuracy on the ImageNet dataset and are widely used as feature extractors.
# 
# ### ResNet (Residual Networks)
# **ResNet**, short for Residual Networks, is a deep learning model architecture introduced by Kaiming He and his colleagues at Microsoft Research. It addresses the issue of vanishing gradients in deep networks by introducing "residual connections," which are shortcuts that skip one or more layers. This allows the network to learn residual functions with reference to the layer inputs, which significantly improves training efficiency and accuracy. ResNet models come in various depths, such as ResNet-50, ResNet-101, and ResNet-152, indicating the number of layers in the network.
# 
# ### MobileNet
# **MobileNet** is a family of lightweight deep learning models designed for efficient image classification on mobile and embedded devices. Developed by Google, MobileNet models use depthwise separable convolutions to reduce the number of parameters and computational cost. This makes them suitable for real-time applications on devices with limited resources. MobileNet architectures include versions like MobileNetV1, V2, and V3, each providing improvements in accuracy and efficiency.
# 
# ---

# ## Quiz: Define and Manipulate a Dictionary for Model Accuracies
# 
# ### Part 1: Define the Dictionary
# Define a dictionary named `model_accuracies` that contains this data:
# 
# |  **Keys** | **Values**  |
# |---|---|
# |  ResNet | 0.91  |
# |  AlexNet | 0.85  |
# |  VGG |  0.88 |
# |  Inception | 0.92  |
# 
# ### Part 2: Calculate the Average Accuracy
# Write code to calculate the average accuracy of the models in the `model_accuracies` dictionary and store it in the variable `average_accuracy`.
# 
# ### Part 3: Find the Best Model
# Write code to find the model with the highest accuracy and store its name in the variable `best_model`.
# 
# ### Part 4: Add a New Model
# Add a new model named `MobileNet` with an accuracy of `0.89` to the `model_accuracies` dictionary.
# 
# ### Instructions:
# 1. Define a dictionary `model_accuracies` where the key is the name of an image classification model (a string) and the associated value is its accuracy (a float).
# 2. Calculate the average accuracy of the models.
# 3. Find the model with the highest accuracy.
# 4. Add a new model to the dictionary.

# In[4]:


# Part 1: Define the dictionary
# TODO: replace None with appropriate code
# Define a dictionary, `model_accuracies`, that provides information
# on the accuracies of different image classification models. 
# The key is the name of a model (a string), and the associated value 
# is its accuracy (a float).
#   Key      |   Value
# ResNet     |   0.91
# AlexNet    |   0.85
# VGG        |   0.88
# Inception  |   0.92
model_accuracies = {"ResNet": 0.91, "AlexNet": 0.85, "VGG": 0.88, "Inception": 0.92}

# Part 2: Calculate the average accuracy
# TODO: replace None with appropriate code
average_accuracy = 0
total_accuracies = 0
for v in model_accuracies.values(): 
    total_accuracies += v
    average_accuracies = total_accuracies/len(model_accuracies)

# Part 3: Find the best model
# TODO: replace None with appropriate code
best_model = max(model_accuracies.values())

# Part 4: Add a new model
# TODO: replace None with appropriate code
# Add the model MobileNet with an accuracy of 0.89
new_model = 'MobileNet'
new_accuracy = 0.89

# Add the new model to the dictionary
model_accuracies[new_model] = new_accuracy

### Notebook grading
model_accuracies_solution = {
    "ResNet": 0.91, 
    "AlexNet": 0.85,
    "VGG": 0.88,
    "Inception": 0.92,
    "MobileNet": 0.89
}

if model_accuracies == model_accuracies_solution:
    print("Nice work defining the dictionary!\n")

average_accuracy_solution = 0.89
if average_accuracy == average_accuracy_solution:
    print("Nice work calculating the average accuracy!\n")
else:
    print(f"Double check your average accuracy calculation. It should be {average_accuracy_solution}.")

best_model_solution = 'Inception'
if best_model == best_model_solution:
    print("Nice work finding the best model!\n")
else:
    print(f"Double check your best model calculation. It should be {best_model_solution}.")


# In[ ]:




