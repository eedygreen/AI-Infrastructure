import ast
import pandas as pd
import numpy as np

recent_r2 = 0.55
recent_sse = 49999

previousscores = pd.read_csv('previousscores_l3demo.csv')

maxversion=previousscores['version'].max()

thisversion = maxversion + 1

new_row_r2 = {'metric': 'r2', 
              'version': thisversion, 
              'score': recent_r2}

new_row_sse = {'metric': 'sse', 
               'version': thisversion, 
               'score': recent_sse}

if recent_r2 > previousscores.loc[previousscores['metric'] == 'r2', 'score'].max():
    new_scores = pd.DataFrame([new_row_r2, new_row_sse])
    previousscores = pd.concat([previousscores, new_scores], ignore_index=True)
    previousscores.to_csv('newscores.csv')

# sse score min(), the lesser (close to zero) the better

if recent_sse < previousscores.loc[previousscores['metric'] == 'sse', 'score'].min():
    new_scores = pd.DataFrame([new_row_r2, new_row_sse])
    previousscores = pd.concat([previousscores, new_scores], ignore_index=True)
    previousscores.to_csv('newscores.csv')
