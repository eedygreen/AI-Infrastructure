#!/usr/bin/env python
# coding: utf-8

# ### Exercise: Feature Summary for Tweet
# 
# In this exercise, you'll concatenate model feature descriptions into a single string for a tweet, ensuring the final string does not exceed 140 characters.
# 
# #### Problem Statement
# 
# Given a list of model feature descriptions, concatenate them into a single string for a tweet. The final string should not exceed 140 characters.
# 
# **Example Input**:
# ```python
# features = [
#     "Feature 1: High accuracy",
#     "Feature 2: Low latency",
#     "Feature 3: Scalability",
#     "Feature 4: Robustness",
#     "Feature 5: Interpretability"
# ]
# ```
# 
# **Instructions**:
# 1. Initialize an empty string `feature_summary`.
# 2. Iterate through the `features` list and concatenate each feature to `feature_summary` with a line break (`\n`) in between.
# 3. Ensure that the final `feature_summary` does not exceed 140 characters.

# In[12]:


features = [
    "Feature 1: High accuracy",
    "Feature 2: Low latency",
    "Feature 3: Scalability",
    "Feature 4: Robustness",
    "Feature 5: Interpretability"
]

feature_summary = ""
# Write your loop here
# TODO
for feature in features:
    if len(feature_summary) + len(feature) + 1 > 140:
        if feature_summary:
            feature_summary += "\n"
        feature_summary += feature[:140 - len(feature_summary)]
        break
    if feature_summary:
        feature_summary += "\n"
    feature_summary += feature
    print(feature_summary)
            

# Print the resulting summary
print(feature_summary)

### Notebook grading
def get_solution(features):
    feature_summary = ""
    for feature in features:
        if len(feature_summary) + len(feature) + 1 > 140:
            if feature_summary:
                feature_summary += "\n"
            feature_summary += feature[:140 - len(feature_summary)]
            break
        if feature_summary:
            feature_summary += "\n"
        feature_summary += feature

    return feature_summary

correct_ans = get_solution(features)

if feature_summary == correct_ans:
    print("Well done!")
else:
    print("Make sure you're inserting spaces between each feature, and that the summary is exactly 140 characters long.")


# In[ ]:




