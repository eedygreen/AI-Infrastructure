#!/usr/bin/env python
# coding: utf-8

# ### Inheritance Exercise: Machine Learning Models
# 
# In this exercise, you'll work with a `Model` parent class and several child classes representing different types of machine learning models. You'll implement methods to evaluate the models' performance, manage hyperparameters, and generate synthetic data for training.
# 
# #### Objectives:
# - Understand and implement basic class inheritance and polymorphism in Python.
# - Learn how to evaluate model performance using statistical measures.
# - Generate synthetic data for training and testing machine learning models.
# 
# #### Instructions:
# 
# 1. **Enhance the `Model` Class:**
#    - Add a method `calculate_accuracy` to calculate the accuracy of the model given predictions and true labels.
#    - Add a method `calculate_f1_score` to calculate the F1 score of the model given predictions and true labels.
#    - Add a method `generate_synthetic_data` that generates synthetic data following a Gaussian distribution.
# 
# 2. **Implement Specific Model Classes:**
#    - Create classes `DecisionTree`, `SVM`, and `NeuralNetwork` inheriting from `Model`.
#    - Each class should have specific attributes representing model parameters (e.g., `max_depth` for `DecisionTree`, `kernel` for `SVM`, `layers` for `NeuralNetwork`).
# 
# 3. **Implement the `EnsembleModel` Class:**
#    - Create the `EnsembleModel` class inheriting from `Model`.
#    - Include an attribute `models` which is a list of individual model instances.
#    - Add a method `calculate_ensemble_accuracy` to calculate the accuracy of the ensemble model using majority voting.
# 
# 4. **Unit Tests:**
#    - Ensure all methods are correctly implemented and tested.
# 

# In[1]:


import unittest
import random
import math
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class Model:
    
    def __init__(self, name):
        self.name = name
    
    def calculate_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def calculate_f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')
    
    def generate_synthetic_data(self, mean, std_dev, num_samples, num_features):
        return np.random.normal(mean, std_dev, (num_samples, num_features))
    
class DecisionTree(Model):
    
    def __init__(self, name, max_depth):
        super().__init__(name)
        self.max_depth = max_depth
    
class SVM(Model):
    
    def __init__(self, name, kernel):
        super().__init__(name)
        self.kernel = kernel

class NeuralNetwork(Model):
    
    def __init__(self, name, layers):
        super().__init__(name)
        self.layers = layers
    
class EnsembleModel(Model):
    
    def __init__(self, name, models):
        super().__init__(name)
        self.models = models
    
    def calculate_ensemble_accuracy(self, y_true, y_preds_list):
        # Majority voting
        y_preds = np.array(y_preds_list)
        y_pred_majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=y_preds)
        return self.calculate_accuracy(y_true, y_pred_majority)

# Unit tests to check your solution

class TestModelClass(unittest.TestCase):
    def setUp(self):
        self.model = Model('BaseModel')
        self.decision_tree = DecisionTree('DecisionTree', 10)
        self.svm = SVM('SVM', 'linear')
        self.neural_network = NeuralNetwork('NeuralNetwork', [64, 32, 10])
        self.ensemble = EnsembleModel('Ensemble', [self.decision_tree, self.svm, self.neural_network])
        self.y_true = [0, 1, 1, 0, 1]
        self.y_pred_tree = [0, 1, 1, 0, 0]
        self.y_pred_svm = [1, 1, 0, 0, 1]
        self.y_pred_nn = [0, 1, 1, 1, 1]
        
    def test_initialization(self): 
        self.assertEqual(self.model.name, 'BaseModel', 'Model name should be BaseModel')
        self.assertEqual(self.decision_tree.max_depth, 10, 'Incorrect max_depth for DecisionTree')
        self.assertEqual(self.svm.kernel, 'linear', 'Incorrect kernel for SVM')
        self.assertEqual(self.neural_network.layers, [64, 32, 10], 'Incorrect layers for NeuralNetwork')
        
    def test_calculate_accuracy(self):
        self.assertEqual(self.model.calculate_accuracy(self.y_true, self.y_pred_tree), 0.8, 'Accuracy calculation is incorrect')
        
    def test_calculate_f1_score(self):
        self.assertAlmostEqual(self.model.calculate_f1_score(self.y_true, self.y_pred_tree), 0.8, places=1, msg='F1 score calculation is incorrect')
        
    def test_generate_synthetic_data(self):
        mean = 0
        std_dev = 1
        num_samples = 100
        num_features = 5
        data = self.model.generate_synthetic_data(mean, std_dev, num_samples, num_features)
        self.assertEqual(data.shape, (100, 5), 'Generated synthetic data shape is incorrect')
        
    def test_calculate_ensemble_accuracy(self):
        y_preds_list = [self.y_pred_tree, self.y_pred_svm, self.y_pred_nn]
        print(self.ensemble.calculate_ensemble_accuracy(self.y_true, y_preds_list))
        self.assertAlmostEqual(self.ensemble.calculate_ensemble_accuracy(self.y_true, y_preds_list), 1.0, places=1, msg='Ensemble accuracy calculation is incorrect')

tests = TestModelClass()

tests_loaded = unittest.TestLoader().loadTestsFromModule(tests)

unittest.TextTestRunner().run(tests_loaded)


# In[ ]:




