# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer

---

## Project Description

This project builds a production-grade machine learning pipeline to identify credit card customers who are most likely to churn. Early identification of at-risk customers allows a business to take a proactive retention strategy before losing them.

The pipeline covers the full ML lifecycle:
- **Exploratory Data Analysis (EDA)**: Distributions, correlations, and density transactions, visualized and saved as images to `images/eda/`.
- **Feature Engineering**: categorical encoding and train/test splitting.
- **Model Training**: Random Forest (Tuned via RandomizedSearchCV) and Logistic Regression trained and evaluated in parallel and saved to `models`.
- **Model Evaluation**: Classification reports, ROC curves, and feature importance plots saved to `images/results/`.
- **Model Persistence**: Trained models are saved as `.pkl` files in the `models/` directory for future use.
- **Logging**: All steps and results are logged to `logs/churn_library_YY-MM-DD.log`

Both a **Random Forest Classifier** and a **Logistic Regression** model are trained and compared, with the best Random Forest configuration selected automatically via 3-fold cross-validated randomized search.
---

## Files and Data Description

### Main Files

- `churn_library.py`  
  Core library containing all pipeline functions: data ingestion, EDA, feature engineering, encoding, model training, evaluation and persistence. Designed as a reusable module — functions can be imported independently or run end-to-end via `main()`.

- `churn_script_logging_and_tests.py`  
  Test suite for every function in `churn_library.py`, ensuring correctness and robustness. Also sets up logging configuration to capture detailed logs of the pipeline execution and are wrtten to `logs/churn_library_YY-MM-DD.log` with timestamps for auditability and traceability.

- `churn_notebook.ipynb`  
  Exploratory notebook used during development for data exploration, feature engineering experimentation, and model prototyping. Contains original inline implementation that was subsequently refactored into `churn_library.py` Useful for step-by-step inspection of intermeidate outputs and dats shapes.

---

### Data

- `data/bank_data.csv`  
  Credit card customer dataset sourced from Kaggle. Contains 10,127 rows and 23 columns including demographic, behavioural, and account features.

  | Column | Type | Description |
  |---|---|---|
  | `Attrition_Flag` | categorical | Target — `"Attrited Customer"` or `"Existing Customer"` |
  | `Customer_Age` | numeric | Age of the customer |
  | `Gender` | categorical | M / F |
  | `Education_Level` | categorical | Graduate, High School, etc. |
  | `Marital_Status` | categorical | Married, Single, etc. |
  | `Income_Category` | categorical | Annual income band |
  | `Card_Category` | categorical | Blue, Silver, Gold, Platinum |
  | `Total_Trans_Amt` | numeric | Total transaction amount (last 12 months) |
  | `Total_Trans_Ct` | numeric | Total transaction count (last 12 months) |
  | `Avg_Utilization_Ratio` | numeric | Average card utilization ratio |

  The binary target column `Churn` is derived from `Attrition_Flag` during preprocessing:
```python
  df["Churn"] = df["Attrition_Flag"].apply(
      lambda val: 0 if val == "Existing Customer" else 1
  )
```
---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

---

## Running the Files

### 1. Run the Pipeline

```bash
python churn_library.py
```

Executes the full ML pipeline end-to-end, from data loading to model training and evaluation. Generates EDA plots, trains both models, evaluates them, saves results and logs all steps to a timestamped log file in `logs/`.

1. Creates necessary directories if they don't exist (`images/eda/`, `images/results/`, `models/`)
2. Loads `data/bank_data.csv`
3. Runs EDA and saves plots to `images/eda/`
4. Encodes categorical features and splits data 70/30 between training and testing sets
5. Trains Random Forest (RandomSearchCV, 3-fold CV) and Logistic Regression
6. Saves classification reports, ROC curves, and feature importance to `images/results/`
7. Persists both trained models to `models/`


---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py -v
```

Runs the full test suite against every pipeline function. Each test:
- Calls the target function with real data
- Asserts expected outputs exist (files, columns, shapes)
- Logs `INFO` on success and `ERROR` on failure to `logs/churn_library_YY-MM-DD.log` with timestamps for traceability. The `-v` flag enables verbose output to the console for test results.

---

## Expected Outputs
Minimum expected outputs:

### Models

| File | Description |
|---|---|
| `models/random_forest_model.pkl` | Best Random Forest estimator from RandomizedSearchCV |
| `models/logistic_regression_model.pkl` | Trained Logistic Regression model |

### EDA Images (`images/eda/`)
| File | Description |
|---|---|
| `churn_distribution.png` | Histogram of churn vs non-churn counts |
| `customer_age_distribution.png` | Distribution of customer ages |
| `marital_status_distribution.png` | Normalised bar chart of marital status |
| `total_trans_ct_density.png` | KDE density plot of total transaction count |
| `correlation_heatmap.png` | Heatmap of numeric feature correlations |

### Result Images (`images/results/`)
| File | Description |
|---|---|
| `roc_curves.png` | Overlaid ROC curves for both models |
| `feature_importance.png` | Ranked feature importances from Random Forest |
| `random_forest_classification_report.png` | Train and test classification report — RF |
| `logistic_regression_classification_report.png` | Train and test classification report — LR |

### Logs
| File | Description |
|---|---|
| `logs/churn_library_YY-MM-DD.log` | Timestamped INFO and ERROR entries for every pipeline stage and test |

---

## Notes

- **Logging** — all logs are written in append (mode="a") mode per run to accumulate logs across runs.
- **Dataset assumption** — the pipeline assumes `Attrition_Flag` is present in the raw CSV. If the column name differs, update the lambda in `perform_eda` and `perform_feature_engineering` functions in `churn_library.py` accordingly.
- **Multiprocessing on Linux** — if you encounter `ChildProcessError` from `RandomizedSearchCV(n_jobs=-1)` on Python 3.13, ignore it, it is a warning from the multiprocessing library and does not affect the model training. To suppress it, set `n_jobs=1` in `train_models` function in `churn_library.py`.

- **Class imbalance** — the dataset is imbalanced (~16% churn rate). The current implementation applied class weighting.

Why this is a problem:
A model trained on this data can achieve 84% accuracy by simply predicting "not churned" for every single customer — without learning anything meaningful. It will look good on accuracy but completely fail at the one thing it exists to do: catch churners.

 For production use, consider `class_weight="balanced"` in both estimators.
```
random_forest_classifier = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"   # adds this
)

logistic_regression_classifier =Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(
      max_iter=3000,
      random_state=42,
      class_weight="balanced"   # adds this
    ))
])
```
