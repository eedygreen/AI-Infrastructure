# Roadmap

- [x] Phase 0 — Set up the environment and install dependencies `tasks/001-setup.md`
- [x] Phase 1 — Load Data and Exploratory data analysis and feature engineering `tasks/002-load_explore.md`
- [x] Phase 2 — Preprocess the data and split into training, validation and testing sets (60/20/20) `tasks/003-preprocessing.md`
- [x] Phase 3 — Design and implement the neural network architecture using pytorch `tasks/004-design_the_model.md`
- [x] Phase 4 — Train the model and evaluate its performance using appropriate metrics  `tasks/005-train_model.md`
- [x] Phase 5 — Evaluate the model and Test the model on unseen data and evaluate its generalization (accuracy, precision, recall, F1-score) `tasks/006-evaluate_model.md`
- [x] Phase 6 — Improve and Tune model hyperparameters and architecture to improve performance `tasks/007-model_tuning.md`

current status: Phase 6 completed

## Phase 6 conclusion

Ten configurations were compared on the same 60/20/20 split, all with Adam: a dropout-free baseline (64-32, lr 0.001), dropout 0.1/0.2/0.3, lr 0.01 and 0.0001, three architectures (32; 32-16; 64-32-16) and a combined model. A follow-up experiment then combined dropout 0.3 with early stopping and a tuned decision threshold.

| Model | Accuracy | Precision | Recall | F1 | AUC | Train/val loss gap |
|---|---|---|---|---|---|---|
| Baseline | 0.7257 | 0.7178 | 0.7564 | 0.7366 | 0.7939 | 0.118 |
| Dropout 0.3 (best test F1) | 0.7447 | 0.7157 | 0.8238 | 0.7660 | 0.8178 | 0.009 |
| Combined: 32 units + dropout 0.3 + lr 0.0001 | 0.7483 | 0.7256 | 0.8099 | 0.7654 | 0.8197 | 0.007 |
| Dropout 0.3 + early stopping (patience 10, stopped at epoch 26) | 0.7485 | 0.7199 | 0.8250 | 0.7689 | 0.8206 | 0.003 |
| Dropout 0.3 + early stopping + tuned threshold (0.386) | 0.7318 | 0.6785 | 0.8951 | 0.7719 | 0.8206 | 0.003 |

- **Overfitting was the main problem.** The baseline's train/val gap of 0.12 shrank to under 0.02 with dropout 0.2-0.3, and F1 rose by about 0.03. The deepest network (64-32-16) and lr 0.01 overfit the most.
- **Combining the best changes added nothing.** The combined model ties dropout 0.3 on F1 (0.7654 vs 0.7660) with the smallest loss gap and a slightly higher AUC. The changes address the same problem, so they do not stack.
- **Recommended model:** dropout 0.3 (64-32, lr 0.001): best test F1 (0.7660) and validation F1 (0.7688) of the dropout, learning-rate and architecture experiments, a train/val gap of 0.009 and the highest recall (0.8238). The combined model is a close alternative but does not beat it. Lowering the decision threshold to 0.3 raises recall further at the cost of precision (see the threshold analysis in the notebook).
- **Dropout 0.3 + early stopping + tuned threshold.** These three levers act on different things (regularization, training length, decision rule), unlike the earlier combination. Early stopping (`patience=10`) stopped at epoch 26 and restored the epoch-16 weights: the train/val gap fell from 0.009 to 0.003, AUC rose from 0.8178 to 0.8206 and F1 from 0.7660 to 0.7689, in a quarter of the epochs (single seed; small gains). The threshold was chosen on the validation set (highest precision with recall >= 0.90, giving 0.386). On the test set it raised recall from 0.8250 to 0.8951 and F1 to 0.7719, cutting false negatives from 871 to 522 while false positives rose from 1,597 to 2,110 and accuracy fell by 1.7 points to 0.7318. Recall landed just under the 0.90 target on the test set (0.9001 on validation), so a validation-chosen threshold does not guarantee the target on new data. The clinical team would need to accept the accuracy and false-positive cost.
- **Mission target not met.** All configurations plateau at about 0.75 accuracy, 0.82 AUC and 0.77 F1, well short of the 95-99% accuracy in `docs/mission.md`. Tuning has run out of room; further gains would need new information (feature engineering) rather than more hyperparameter changes.
- **Caveat:** configurations were ranked by test F1 as the task requires. Validation F1 gives the same top group, but the test scores of the selected model are slightly optimistic.
