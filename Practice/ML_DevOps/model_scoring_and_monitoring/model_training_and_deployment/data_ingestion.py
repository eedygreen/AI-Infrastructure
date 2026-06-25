import os
import pandas as pd

directories = ["/data_1/", "/data_2/"]

final_dataframe = pd.DataFrame(columns=['peratio', 'pric'])

for dir in directories:
    filenames = os.listdir(os.getcwd() + dir)
    for file in filenames:
        current_df = pd.read_csv(os.getcwd() + directories + file)
        final_dataframe = pd.concat([final_dataframe, current_df], ignore_index=True)

final_dataframe.to_csv("ingested_data.csv")
