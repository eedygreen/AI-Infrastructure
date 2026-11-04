# Task: Load and Explore Data

## Goal
The goal of this task is to load the diabetes dataset and perform exploratory data analysis (EDA) to understand the structure and characteristics of the data. This will help in identifying any issues, patterns, or insights that can inform the preprocessing and modeling steps.


## Context
Files to read: 
data: `data/diabetes_data.csv`
data dictionary: `data/data_dictionary.md`

## Requirements
1. Function load_data(path) returns a DataFrame. Raise FileNotFoundError if the file does not exist. Use Pandas to read the CSV file and return a DataFrame.
2. Display the DataFrame's shape, data types, and whether any values are missing `missing_values =  # ------- Add your code here`. Use Pandas DataFrames methods that provide a concise summary of this information in a single call.
3. Analyze the target variable distribution. `target_distribution =  # ------- Add your code here`
`target_percentage =  # ------- Add your code here`. Pandas Series (single columns) have methods to count the frequency of each unique value.
4. Examine basic statistical properties of the dataset Statistical summaries (mean, std, min, max, quartiles) help you understand feature distributions and identify potential outliers. DataFrames have built-in methods that compute these statistics across all numerical columns at once.
5. Analyze correlation between features and the target variable . compute pairwise correlations between all features, then focus on correlations between each feature and your target variable. `correlations =  # ------- Add your code here`
6. Write Observations as bullet points. Discuss what you'd need to know before preprocessing the data.


## Constraints
- Use Pandas for data loading and exploration.
- `diabetes_prediction_mlp2.ipynb` at the root of the directory is the file to be modified.
- Use pandas methods to perform the required analysis and do not make assumptions about the data without verifying them through code.

## Deliverables
Ensure you these criteria are met:
  - [ ] The target variable distribution and its implications for evaluation
  - [ ] Which features show strong correlations with diabetes
  - [ ] The need for feature scaling due to different value ranges
  - [ ] The absence of missing values (simplifies preprocessing)

## Report back
- After completing the task, report back with the following:
  - The shape of the DataFrame (number of rows and columns).
  - The data types of each column.
  - Whether there are any missing values in the dataset.
  - Any questions or considerations for preprocessing the data based on your exploration.