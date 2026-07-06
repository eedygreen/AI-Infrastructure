"""
A Module for predicting churn customers

Author: I.Idris
Date: 15/05/2026
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from constants import CATEGORY_COLUMNS, QUANT_COLUMNS

os.environ["QT_QPA_PLATFORM"] = "offscreen"
matplotlib.use("Agg")

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = r"./data/bank_data.csv"


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """

    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """

    figsize = (20, 10)

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # --- 1. Churn distribution ---
    fig, ax = plt.subplots(figsize=figsize)
    df["Churn"].hist(ax=ax)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn Flag")
    ax.set_ylabel("Count")
    fig.savefig(os.path.join(
        EDA_DIR,
        "churn_distribution.png"),
        bbox_inches="tight")
    plt.close(fig)

    # --- 2. Customer age distribution ---
    fig, ax = plt.subplots(figsize=figsize)
    df["Customer_Age"].hist(ax=ax)
    ax.set_title("Customer Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    fig.savefig(os.path.join(
        EDA_DIR,
        "customer_age_distribution.png"),
        bbox_inches="tight")
    plt.close(fig)

    # --- 3. Marital status distribution ---
    fig, ax = plt.subplots(figsize=figsize)
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar", ax=ax)
    ax.set_title("Marital Status Distribution (Normalized)")
    ax.set_xlabel("Marital Status")
    ax.set_ylabel("Proportion")
    fig.savefig(os.path.join(
        EDA_DIR,
        "marital_status_distribution.png"),
        bbox_inches="tight")
    plt.close(fig)

    # --- 4. Total transaction count density ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True, ax=ax)
    ax.set_title("Total Transaction Count Density")
    ax.set_xlabel("Total Transaction Count")
    fig.savefig(os.path.join(
        EDA_DIR,
        "total_trans_ct_density.png"),
        bbox_inches="tight"
        )
    plt.close(fig)

    # --- 5. Correlation heatmap ---
    fig, ax = plt.subplots(figsize=figsize)
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=False, cmap="Dark2_r", linewidths=2, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(os.path.join(
        EDA_DIR,
        "correlation_heatmap.png"),
        bbox_inches="tight"
        )
    plt.close(fig)


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """

    df = df.copy()

    for category in category_lst:
        category_groups = df.groupby(category)[response].mean()
        df[f"{category}_{response}"] = df[category].map(category_groups)

    return df


def perform_feature_engineering(df, response):
    """
    Select model features, encode categorical columns,
    and Split dataset into train and test sets.

    input:
        df: pandas dataframe
        response: response column name
    output:
        x_train, x_test, y_train, y_test
    """
    
    x_train, x_test, y_train, y_test = train_test_split(
        df, df[response], test_size=0.3, random_state=42)
    
    for col in CATEGORY_COLUMNS:
        train_means = x_train.groupby(col)[response].mean()
        x_train[f"{col}_{response}"] = x_train[col].map(train_means)
        x_test[f"{col}_{response}"] = x_test[col].map(train_means)

    x_train = x_train[QUANT_COLUMNS]
    x_test = x_test[QUANT_COLUMNS]

    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Save classification reports as images.

    input:
        y_train:           true labels for training set
        y_test:            true labels for test set
        y_train_preds_lr:  logistic regression predictions on train set
        y_train_preds_rf:  random forest predictions on train set
        y_test_preds_lr:   logistic regression predictions on test set
        y_test_preds_rf:   random forest predictions on test set
    output:
        None
    """

    reports = {
        "Random Forest": {
            "train": classification_report(y_train, y_train_preds_rf),
            "test": classification_report(y_test, y_test_preds_rf),
        },
        "Logistic Regression": {
            "train": classification_report(y_train, y_train_preds_lr),
            "test": classification_report(y_test, y_test_preds_lr),
        },
    }

    for model_name, splits in reports.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")

        report_text = (
            f"{model_name} — Train\n\n"
            f"{splits['train']}\n"
            f"{'-' * 60}\n\n"
            f"{model_name} — Test\n\n"
            f"{splits['test']}"
        )

        ax.text(
            0.01, 0.99,
            report_text,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
        )

        filename = f"{model_name.lower().replace(
            ' ', '_')}_classification_report.png"
        fig.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches="tight")
        plt.close(fig)


def feature_importance_plot(model, x_data, output_pth):
    """
    Compute, plot, and save feature importances for a tree-based model plot.

    input:
        model: fitted model with a best_estimator_ attribute
                (e.g. GridSearchCV) or any tree-based estimator
                exposing feature_importances_
        x_data: pandas dataframe of features used during training
        output_pth: directory path where the plot image will be saved
    output:
        None
    """

    os.makedirs(output_pth, exist_ok=True)

    estimator = getattr(model, "best_estimator_", model)

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(
            f"{type(estimator).__name__} does not expose feature_importances_."
            "Use a tree-based model such as RandomForest or GradientBoosting."
        )
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    fig, ax = plt.subplots(figsize=(20, 5))

    ax.set_title("Feature Importance")
    ax.set_ylabel("Importance")
    ax.bar(range(x_data.shape[1]), importances[indices])
    ax.set_xticks(range(x_data.shape[1]))
    ax.set_xticklabels(names, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_pth, "feature_importance.png"),
        bbox_inches="tight",
    )

    plt.close(fig)


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models, evaluate performance and save outputs,
    classification reports, ROC curves, and features importances.

    input:
        x_train: training features dataframe
        x_test:  test features dataframe
        y_train: training labels series
        y_test:  test labels series

    output:
        None
    """

    # === 1. Define models ===
    random_forest_classifier = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        )
    logistic_regression_classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=3000,
            random_state=42,
            class_weight="balanced",))
    ])

    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini']
    }

    # === 2. Train ===
    cross_validation_rfc = RandomizedSearchCV(
        estimator=random_forest_classifier,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=1,               # Set -1 to use all CPU cores,
        error_score='raise'
    )

    cross_validation_rfc.fit(x_train, y_train)
    logistic_regression_classifier.fit(x_train, y_train)

    # === 3. Predict ===
    y_train_preds_rf = cross_validation_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cross_validation_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = logistic_regression_classifier.predict(x_train)
    y_test_preds_lr = logistic_regression_classifier.predict(x_test)

    # === 4. Classification reports ===
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    )

    # === 5. ROC Curves ===
    _save_roc_curves(cross_validation_rfc.best_estimator_,
                     logistic_regression_classifier, x_test, y_test)

    # === 6. Features importance ===
    feature_importance_plot(cross_validation_rfc,
                            x_train, output_pth=RESULTS_DIR)

    # === 7. Persiste models ===
    joblib.dump(cross_validation_rfc.best_estimator_,
                os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    joblib.dump(logistic_regression_classifier,
                os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))


def _save_roc_curves(rf_model, lr_model, x_test, y_test):
    """
    Plot and save overlaid ROC curves for models

    Input:
        rf_model: fitted RandomForestClassifier
        lr_model: fitted LogisticRegression
        x_test:   test features dataframe
        y_test:   test labels series

    outputs:
        None
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for model, label in [(rf_model, "Radom Forest"),
                         (lr_model, "Logistic Regression")]:
        RocCurveDisplay.from_estimator(
            model, x_test, y_test, ax=ax, name=label
        )

    ax.set_title("ROC Curves - Model Comparison")
    ax.plot([0, 1], [0, 1], linestyle="--",
            color="grey", label="Random Chance")
    ax.legend(loc="lower right")

    fig.savefig(os.path.join(
        RESULTS_DIR, "roc_curves.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_output_directories()

    df = import_data(DATA_PATH)

    perform_eda(df)

    df = encoder_helper(df, CATEGORY_COLUMNS, "Churn")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(x_train, x_test, y_train, y_test)
