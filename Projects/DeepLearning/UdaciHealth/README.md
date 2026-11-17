# UdaciHealth: Diabetes Risk Prediction with a PyTorch MLP

A multi-layer perceptron (MLP) that pre-screens patients for diabetes risk from CDC survey data, so a regional hospital network can prioritise limited diagnostic testing. The full workflow (EDA, preprocessing, model design, training, evaluation and tuning) lives in `diabetes_prediction_mlp.ipynb`.

> **Screening only.** The model does not diagnose, does not guide treatment decisions and does not replace clinical judgement.

## Results at a glance

| Metric (test set) | Baseline (64-32, lr 0.001) | Recommended (dropout 0.3) | Dropout 0.3 + early stopping + tuned threshold |
|---|---|---|---|
| Accuracy | 0.7257 | 0.7447 | 0.7318 |
| Precision | 0.7178 | 0.7157 | 0.6785 |
| Recall | 0.7564 | 0.8238 | 0.8951 |
| F1 | 0.7366 | 0.7660 | 0.7719 |
| ROC AUC | 0.7939 | 0.8178 | 0.8206 |

- **Overfitting was the main problem.** Dropout 0.2–0.3 cut the train/validation loss gap from 0.118 to under 0.02 and raised F1 by about 0.03.
- **Recommended model:** dropout 0.3 on the 64-32 network. It has the best test F1 among the dropout, learning-rate and architecture experiments and the highest recall of that group.
- **Recall is tunable.** Lowering the decision threshold to 0.386 (chosen on the validation set) lifted test recall to 0.895 and cut false negatives from 871 to 522, at the cost of 513 more false positives and 1.7 points of accuracy. The clinical team needs to decide whether that trade is acceptable.
- **The mission's 95–99% accuracy target was not met.** Every configuration plateaus at about 0.75 accuracy, 0.82 AUC and 0.77 F1. Further gains need new information (feature engineering), not more hyperparameter tuning.
- Configurations were ranked by test F1, so the selected model's test scores are slightly optimistic. Validation F1 picks the same top group.

See `docs/roadmap.md` for the full ten-configuration comparison.

## Dataset

- **Source:** [CDC Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (BRFSS 2015), at `data/diabetes_data.csv`.
- **Size:** 50,000 rows, balanced subset of the original 253,680 (about 50.7% positive).
- **Features:** 21 self-reported health, lifestyle and demographic indicators (BMI, blood pressure, age category, general health and so on).
- **Target:** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).
- **Limitations:** self-reported, no lab values (glucose, HbA1c), and prediabetes and diabetes share one positive class. Because the data was downsampled to 50/50, predicted probabilities do not reflect real-world prevalence (about 14%).

Feature definitions and ethical notes are in `data/data_dictionary.md`.

## Method

1. **Split:** stratified 60/20/20 train/validation/test (29,437 / 9,812 / 9,813 rows).
2. **Preprocess:** `StandardScaler` fitted on the training set only, then applied to validation and test; tensors fed through `DataLoader`s with batch size 64.
3. **Model:** `Linear(21→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → Sigmoid` (3,521 parameters), trained with Adam and binary cross-entropy. The recommended model adds `Dropout(0.3)` after each hidden layer.
4. **Tune:** dropout rates, learning rates, three architectures, a combined model, then early stopping (patience 10) and decision-threshold tuning.
5. **Reproducibility:** random seed 42 for NumPy and PyTorch.

## Getting started

Requires Python 3.10.

```sh
git clone <repo-url>
cd UdaciHealth
pip install -r requirements.txt
jupyter lab diabetes_prediction_mlp.ipynb
```

`requirements.txt` pins CUDA 12.1 builds of PyTorch (`torch==2.5.1+cu121`). On a CPU-only machine, install the CPU build of PyTorch first, or remove the `+cu121` suffixes. The notebook uses a GPU when one is available and falls back to CPU otherwise.

## Project structure

```
UdaciHealth/
├── diabetes_prediction_mlp.ipynb   # Main notebook: EDA, training, evaluation, tuning
├── data/
│   ├── diabetes_data.csv           # Dataset
│   └── data_dictionary.md          # Feature descriptions and clinical context
├── docs/
│   ├── mission.md                  # Mission and success criteria
│   ├── roadmap.md                  # Phases and tuning results
│   └── architecture.md
├── tasks/                          # Per-phase implementation tasks (001–007)
├── tests/
├── requirements.txt
└── CLAUDE.md                       # Project instructions for Claude Code
```

## Stakeholders

| Stakeholder | Need |
|---|---|
| Clinical team | Interpretable, actionable results |
| Hospital administrators | Cost-effectiveness |
| IT department | A deployment-ready model |

## Built with

PyTorch, scikit-learn, pandas, NumPy, Matplotlib, Jupyter.
