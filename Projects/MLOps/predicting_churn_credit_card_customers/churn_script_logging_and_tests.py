"""
Module for Testing library and logging the output

Author: I. Idris

Date: 17/05/2026
"""

import os
import churn_library as cls
from utils import logger
from constants import CATEGORY_COLUMNS

log = logger

DATA_PATH = "./data/bank_data.csv"


def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)
        log.info("Data imported successfully!")

    except FileNotFoundError as err:
        log.error(f"Data import failed: {err}")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        log.error(f"Test failed with assertion error: {err}")
        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)
        log.info("EDA completed successfully")

    except Exception as err:
        log.error(f"EDA failed: {err}")

    expected_plots = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "total_trans_ct_density.png",
        "correlation_heatmap.png",
    ]

    for plot in expected_plots:
        try:
            assert os.path.isfile(os.path.join(cls.EDA_DIR, plot))
            log.info(f"EDA plot found: {plot}")

        except AssertionError as err:
            log.error(f"Expected EDA plot missing: {plot}")
            raise err


def test_encoder_helper(encoder_helper):
    """Test one-hot encoding and the means that fill cateogry columns."""

    try:
        df = cls.import_data(DATA_PATH)

        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        df = encoder_helper(df, CATEGORY_COLUMNS, response="Churn")

        assert "Gender_Churn" in df.columns
        assert "Education_Level_Churn" in df.columns
        assert "Marital_Status_Churn" in df.columns
        assert "Income_Category_Churn" in df.columns
        assert "Card_Category_Churn" in df.columns

        log.info("Encoder helper completed successfully")

    except AssertionError as err:
        log.error(f"Encoded column missing in daatframe: {err}")
        raise err

    except Exception as err:
        log.error(f"Encoded helper failed: {err}")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)

        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, response="Churn")

        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]

        log.info("Feature: test completed successfully!")

    except AssertionError as err:
        log.error(
            f"Feature: test failed with assertion error: {err}")
        raise err

    except Exception as err:
        log.error(f"Feature: test failed: {err}")
        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)

        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df, response="Churn"
        )

        train_models(x_train, x_test, y_train, y_test)

        assert os.path.isfile(os.path.join(cls.MODELS_DIR,
                                           "random_forest_model.pkl"))
        assert os.path.isfile(os.path.join(cls.MODELS_DIR,
                                           "logistic_regression_model.pkl"))

        assert os.path.isfile(os.path.join(cls.RESULTS_DIR, "roc_curves.png"))
        assert os.path.isfile(
            os.path.join(cls.RESULTS_DIR,
                         "feature_importance.png"))
        assert os.path.isfile(
            os.path.join(cls.RESULTS_DIR,
                         "random_forest_classification_report.png"))
        assert os.path.isfile(
            os.path.join(cls.RESULTS_DIR,
                         "logistic_regression_classification_report.png"))
        log.info("Model training: test completed successfully!")
    except AssertionError as err:
        log.error(f"Model training: output file missing: {err}")
        raise err
    except Exception as err:
        log.error(f"Model training failed: {err}")
        raise err


if __name__ == "__main__":

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
