# Task: Preprocessing Data

## Goal
The goal of this task is to implement a preprocessing function that prepares the diabetes dataset for training a machine learning model. This includes handling missing values, scaling features, and splitting the data into training and testing sets.


## Context
df: DataFrame containing the diabetes dataset


## Requirements
1. Split the DataFrame into features (X) and target (y). The target variable is 'diabetes'.

2. Scale the features using StandardScaler from sklearn.preprocessing. Fit the scaler on the training data and transform both the training and testing data.

3. Split data intor three sets(training, validation, and testing): 
    create a Function `preprocess_data(dataframe)` returns a DataFrame. Raise ValueError if the DataFrame is empty. Use Pandas to process the data and return a DataFrame.
    a. Split the data into training, validation and testing sets (60/20/20) using train_test_split from sklearn.model_selection. Use 60% of the data for training, 20% for validation, and 20% for testing. Set random_state=RANDOM_SEED variable for reproducibility.
    You'll need to split in two stages: first separate training from temp, then split temp into validation and test. Each split should maintain the same diabetes/no-diabetes ratio as the original dataset. Look for a parameter that handles stratification.

    b. Return the training and testing sets as a tuple: (X_train, X_test, X_val, y_train, y_test, y_val)

    c. Ensure that the training, validation, and testing sets have the same diabetes/no-diabetes ratio as the original dataset. Use stratified sampling to achieve this.
    - Training set contains approximately 60% of the data
    - Validation and test sets each contain approximately 20% of the data
    - Class presence is as expected (same ratio as for the full dataset) in all three splits

  If they aren’t, revisit your splitting logic before moving forward.

4. Normalize the features using StandardScaler from sklearn.preprocessing. Fit the scaler on the training data and transform both the training, validation, and testing data. follow these steps:
    a. Create an instance of StandardScaler.
    b. Fit the scaler on the training features (X_train).
    c. Transform the training, validation, and testing features using the fitted scaler.
create a function `scale_features(X_train, X_val, X_test)` that returns the scaled features as a tuple: (X_train_scaled, X_val_scaled, X_test_scaled)

5. Convert the scaled Numpy arrays to Pytorch tensors. Features should be floating-point tensors for neural network computations, while binary classification targets work with long tensors. PyTorch provides tensor creation functions for different data types. expected variables: X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor 

6. Create a DataLoader for the training, validation, and testing sets. PyTorch provides utilities in torch.utils.data to wrap tensors into datasets and create batching iterators.
You'll need to: (1) combine feature and label tensors into dataset objects, and 
(2) wrap datasets in loaders with appropriate batch sizes. Training data should be shuffled, but validation/test should not.
Use torch.utils.data.TensorDataset to create datasets from the tensors, and then use torch.utils.data.DataLoader to create data loaders for each dataset. Set batch_size=64 and shuffle=True for the training DataLoader, while shuffle=False for the validation and testing DataLoaders. expected variables: train_loader, val_loader, test_loader

7. Validate DataLoader outputs. Confirm that batches are correctly structured and preprocessed:
  - **Feature batch shape:** `(64, 21)` → 64 samples × 21 features  
   - **Label batch shape:** `(64,)` → 64 labels 
  - **Feature values:** normalized 
  - **Labels:** binary values (**0 or 1**)  
  
  If any shapes or values look off, revisit step 6.


## Constraints
- Avoid using loops for data processing; leverage vectorized operations and built-in functions for efficiency.
- Avoid hardcoding values; use dynamic calculations based on the dataset's properties to ensure flexibility and adaptability to different datasets.
- Avoid switching back and forth between Numpy arrays and Tensors: each conversion moves data between memory spaces and slows down processing.

## Deliverables
Ensure you these criteria are met:
  - [ ] Class presence is as expected (same ratio as for the full dataset) in all three splits
  - [ ] Validation and test sets each contain approximately 20% of the data 
  - [ ] Training set contains approximately 60% of the data
  - [ ] **Feature batch shape:** `(64, 21)` → 64 samples × 21 features
  - [ ] **Label batch shape:** `(64,)` → 64 labels 
  - [ ] **Feature values:** normalized 
  - [ ] **Labels:** binary values (**0 or 1**) s (simplifies preprocessing)

## Report back
- After completing the task, report back with the following:
  - The shape of the batches.
  - The Label batch shape.
  - The normalized feature values.
  - Any questions or considerations for designing the neural network.