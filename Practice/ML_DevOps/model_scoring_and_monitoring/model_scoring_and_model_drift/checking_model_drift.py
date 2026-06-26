import ast
import numpy as np

# Define a hypothetical new F1 score:
newf1 = 0.38

# previous f1_scores
with open('previousscores_l3demo.txt', 'r') as f:
    f1list = ast.literal_eval(f.read())

# Raw Comparison Test
firsttest = newf1 < np.min(f1list)
print(firsttest)

# Parametric Significance Test
secondtest = newf1 < np.mean(f1list) - 2 * np.std(f1list)
print(secondtest)

# Non Parametric Significance Test
iqr = np.quantile(f1list, 0.75) - np.quantile(f1list, 0.25)
thirdtest = newf1 < np.quantile(f1list, 0.25) - iqr * 1.5
print(thirdtest)
