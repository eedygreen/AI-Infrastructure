import json
import os
import pickle
import subprocess
import sys
import timeit

import pandas as pd
import urllib.request
from importlib.metadata import version, PackageNotFoundError


################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

FEATURE_COLUMNS = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COLUMN = 'exited'


################## Function to get model predictions
def model_predictions(test_data):
    # Read the deployed model and a dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(test_data[FEATURE_COLUMNS])
    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary():
    # Calculate summary statistics here
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    numeric_data = data.select_dtypes(include=['number'])

    means = numeric_data.mean().tolist()
    medians = numeric_data.median().tolist()
    modes = numeric_data.mode().iloc[0].tolist()

    return [means, medians, modes]


################## Function to get percent of missing data
def missing_data():
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    return (data.isna().mean() * 100).tolist()


##################Function to get timings
def execution_time():
    # Calculate timing of ingestion.py and training.py
    start_time = timeit.default_timer()
    subprocess.run([sys.executable, 'ingestion.py'], check=True)
    ingestion_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    subprocess.run([sys.executable, 'training.py'], check=True)
    training_time = timeit.default_timer() - start_time

    return [ingestion_time, training_time]


################## Function to check dependencies
def outdated_packages_list():
    # Get a list of current and latest package versions
    requirements = {}
    try:

        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '==' not in line:
                    continue
                package = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                requirements[package] = None
    except FileNotFoundError:
        return pd.DataFrame()


    rows = []
    for package in requirements.keys():
        try:
            installed_version = version(package)
        except PackageNotFoundError:
            installed_version = "not installed"
        
        latest_version = get_latest_version(package)
        
        rows.append({
            'package': package,
            'current_version': installed_version,
            'latest_version': latest_version,
            'is_outdated': installed_version != latest_version and latest_version != "unkown"
        })

    return pd.DataFrame(rows)

def get_latest_version(package_name):
    # Get the latest stable version from pypi
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        # Use a user-agent to avoid 403 errors from PyPI
        req = urllib.request.Request(url, headers={"User-Agent": "Python/urllib"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data["info"]["version"]
    except Exception as e:
        print(f"geting latest version Error: {e}")
        return "unknown"


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    print(f"model prediction: {model_predictions(data)}")
    print(f"Data Summary: {dataframe_summary()}")
    print(f"Missing Data: {missing_data()}")
    print(f"Execution Time: {execution_time()}")
    print(f"Outdated Packages:\n {outdated_packages_list()}")
