import os
import argparse
import logging
import json

import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(project="decision_tree", job_type="train")

    logger.info("Downloading and reading test artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    logger.info("Setting up pipeline")

    pipe = get_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train, y_train)

    logger.info("Scoreing")
    score = roc_auc_score(
        y_val, pipe.predict_proba(X_val), average="macro", multi_class="ovo"
    )

    run.summary['AUC'] = score

    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )

    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]

    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])

    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "titler + song_names")

    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0])),
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)

    fig_feat_imp.tight_layout()

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))

    y_pred = pipe.predict(X_val)

    cm = confusion_matrix(
        y_true=y_val,
        y_pred=y_pred,
        labels=pipe["classifier"].classes_,
        normalize="true"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=pipe["classifier"].classes_
    )

    disp.plot(
        ax=sub_cm,
        values_format=".1f",
        xticks_rotation=90,
    )

    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )


def get_inference_pipeline(args):

    categorical_feaures = sorted(["time_signature", "key"])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        OrdinalEncoder()
    )

    numerical_features = sorted([
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ])
    numerical_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    nlp_features = ["text_feature"]
    reshape_to_1d = FunctionTransformer(lambda x: x.reshape(-1), validate=False)
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(binary=True),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_feaures),
            ('nlp1', nlp_transformer, nlp_features),
        ],
        remainder="drop", # This drops the columns that we do not transform
    )

    with open(args.model_config) as fp:
        model_config = json.load(fp)
    wandb.config.update(model_config)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config)),
        ]
    )
    return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualifies name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a JSON file containing the model configuration",
        required=True
    )

    args = parser.parse_args()

    go(args)
