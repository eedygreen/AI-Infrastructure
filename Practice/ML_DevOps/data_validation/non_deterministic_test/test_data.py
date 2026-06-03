import pytest
import wandb
import pandas as pd
import scipy.stats


run = wandb.init(project="non_deterministic", job_type="data_tests")


@pytest.fixture(scope="session")
def data():
    local_path = run.use_artifact("idris-isah2-udacity/preprocess/data_train.csv:v0").file()
    sample_1 = pd.read_csv(local_path)

    local_path = run.use_artifact("idris-isah2-udacity/preprocess/data_test.csv:v0").file()
    sample_2 = pd.read_csv(local_path)

    return sample_1, sample_2


def test_kolmogorov_smirnov(data):
    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    alpha = 0.05
    alpha_prime = 1 - (1 - alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:
        ts, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )
        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime