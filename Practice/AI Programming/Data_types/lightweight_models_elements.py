#!/usr/bin/env python
# coding: utf-8

# ## Lightweight Models in Deep Learning
# 
# **Lightweight models** are deep learning models designed to be efficient in terms of memory usage, computational cost, and power consumption. These models are particularly useful for deployment on devices with limited resources, such as mobile phones, embedded systems, and edge devices. The primary goals of lightweight models are to reduce the number of parameters and operations required for inference while maintaining a high level of accuracy.
# 
# Lightweight models play a crucial role in making AI accessible and practical for a wide range of applications, especially in scenarios where computational resources are limited.
# 
# Here are some key characteristics and benefits of lightweight models:
# 
# 1. **Reduced Parameters and Computations:**
#    - Lightweight models have fewer parameters compared to traditional deep learning models. This reduction is achieved through techniques like depthwise separable convolutions and pruning.
#    - Fewer parameters lead to fewer computations, making the models faster and more efficient.
# 
# 2. **Lower Memory Footprint:**
#    - These models require less memory for storing parameters and intermediate feature maps, making them suitable for deployment on devices with limited RAM.
# 
# 3. **Energy Efficiency:**
#    - Lightweight models consume less power, which is crucial for battery-powered devices like smartphones and IoT devices.
# 
# 4. **Faster Inference:**
#    - Due to reduced complexity, lightweight models can perform inference faster, enabling real-time applications such as augmented reality, real-time video analysis, and on-device AI.
# 
# ### Examples of Lightweight Models
# 
# #### MobileNet
# **MobileNet** is a family of lightweight deep learning models developed by Google. MobileNet models use depthwise separable convolutions to reduce the number of parameters and computational cost. This makes them suitable for real-time applications on devices with limited resources. MobileNet architectures include versions like MobileNetV1, V2, and V3, each providing improvements in accuracy and efficiency.
# 
# #### SqueezeNet
# **SqueezeNet** is another lightweight model designed to achieve AlexNet-level accuracy with significantly fewer parameters. It introduces "fire modules," which consist of squeeze layers (with 1x1 filters) and expand layers (with both 1x1 and 3x3 filters). SqueezeNet is highly efficient, making it suitable for deployment on devices with limited computational resources.
# 
# #### EfficientNet
# **EfficientNet** is a family of convolutional neural network (CNN) architectures and scaling methods developed by Google. EfficientNet optimizes both the network architecture and scaling methods by using a compound scaling method that uniformly scales the depth, width, and resolution of the network. This approach achieves high performance with fewer parameters and lower computational cost compared to traditional models.
# 
# ---
# 

# ## Quiz: Adding Properties to Nested Dictionaries for Image Classification Models
# 
# In this exercise, you will work with nested dictionaries. Add another entry, `is_lightweight`, to each dictionary in the `models` dictionary. After inserting the new entries, you should be able to perform these lookups:
# 
# ```python
# >>> print(models['ResNet']['is_lightweight'])
# False
# >>> print(models['MobileNet']['is_lightweight'])
# True
# ```
# 
# Here is the initial `models` dictionary:

# In[1]:


models = {
    'ResNet': {'layers': 50, 'accuracy': 0.91, 'type': 'CNN', 'is_lightweight': False},
    'MobileNet': {'layers': 28, 'accuracy': 0.89, 'type': 'CNN', 'is_lightweight': True}
}

# TODO: Add an 'is_lightweight' entry to the ResNet and MobileNet dictionaries
# hint: MobileNet is a lightweight model, ResNet isn't


### Notebook grading
explanation_str = '''Your code produced the wrong result. Looks like you did not add the 'is_lightweight' property properly for {}.'''
if not('is_lightweight' in models['ResNet'].keys()) or models['ResNet']['is_lightweight'] != False:
    print(explanation_str.format("'ResNet'"))
elif not('is_lightweight' in models['MobileNet'].keys()) or models['MobileNet']['is_lightweight'] != True:
    print(explanation_str.format("'MobileNet'"))
else:
    print("""Your code passes all of our tests, nice work!""")


# In[ ]:




