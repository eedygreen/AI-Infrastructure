import argparse
import logging
import os
import pandas as pd
import numpy as np
import wandb
import tempfile
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project='preprocess', job_type='process_data')

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    if args.stratify != "null" and args.stratify in df.columns:
        df = df.dropna(subset=[args.stratify])

    numerical_columns = [
       "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]

    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=[c for c in numerical_columns if c in df.columns])

    logger.info("Splitting data into train, val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != "null" else None,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split, df in splits.items():

            artifact_name = f"{args.artifact_root}_{split}.csv"

            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            df.to_csv(temp_path, index=False)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            artifact.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a datase",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
            "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the artifact",
        required=True,
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="If set, it is the name of a column to yse for stratified splitting",
        required=False,
        default='null'
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Fraction of dataset or number of items to include in splitting",
        required=True
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="An integer number to use to init the random number generator. " \
            "It ensures repeatibility in the splitting",
        required=False,
        default=42
    )

    args = parser.parse_args()

    go(args)
