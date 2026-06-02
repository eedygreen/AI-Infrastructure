import pandas as pd
import wandb
import os

run = wandb.init(project='preprocess', job_type='preprocess_data')

artifact = run.use_artifact("idris-isah2-udacity/EDA/genres_mod.parquet:v0")
artifact_dir = artifact.download()

df = pd.read_parquet(os.path.join(artifact_dir, "genres_mod.parquet"))

df = df.drop_duplicates().reset_index(drop=True)

# New text features
df['title'].fillna(value='', inplace=True)
df['song_name'].fillna(value='', inplace=True)
df['text_feature'] = df['title'] + ' ' + df['song_name']

# save to file
df.to_csv("preprocessed_data.csv", index=False)

artifact = wandb.Artifact(
    name="preprocessed_data.csv",
    type="dataset",
    description="Preprocessed dataset with duplicates removed and features added"
)

artifact.add_file("preprocessed_data.csv")
run.log_artifact(artifact)

run.finish()
