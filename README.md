# CS135 Project A — Ready-to-Run Baselines (Problem 1 and 2)

This folder contains two Python scripts that match the assignment requirements and produce the artifacts you need (plots, JSON of best params, and leaderboard files). You must understand the code and write your report yourself. Cite any external code you adapt.

## Folder layout

```
tufts_cs135_projectA/
  projectA_p1_bow_logreg.py
  projectA_p2_bert_baselines.py
  artifacts/                 # create your own output dirs per problem
```

## Data expectation

Place the course-provided data under `./data_readinglevel/` (same directory where you run the scripts):

- `x_train.csv`, `y_train.csv`, `x_test.csv`
- (Problem 2) `x_train_BERT_embeddings.npz`, `x_test_BERT_embeddings.npz`

## Problem 1 (BoW + Logistic Regression)

Run:

```bash
python projectA_p1_bow_logreg.py --data_dir ./data_readinglevel --out_dir ./artifacts_p1
```

Outputs:

- `artifacts_p1/p1_best_params.json`
- `artifacts_p1/p1_hyperparam_curve.png` (training vs validation AUROC vs C)
- `artifacts_p1/p1_confusion_matrix.png` (one fold, threshold=0.5)
- `yproba1_test.txt` in current working directory (for Gradescope leaderboard)

Notes:
- Uses StratifiedGroupKFold grouped by author to reduce leakage. If unavailable, falls back to StratifiedKFold.
- Hyperparameters searched: BoW (binary, min_df, max_df, ngram_range) and LR (penalty, C). Scoring is AUROC.

## Problem 2 (Embeddings + flexible classifiers)

Run (logistic on BERT embeddings):

```bash
python projectA_p2_bert_baselines.py --data_dir ./data_readinglevel --out_dir ./artifacts_p2 --clf logistic
```

Alternative classifiers:

```bash
# Linear SVM (calibrated to produce probabilities)
python projectA_p2_bert_baselines.py --data_dir ./data_readinglevel --out_dir ./artifacts_p2 --clf linear_svm

# MLP
python projectA_p2_bert_baselines.py --data_dir ./data_readinglevel --out_dir ./artifacts_p2 --clf mlp
```

Outputs:

- `artifacts_p2/p2_best_params.json`
- `artifacts_p2/p2_hyperparam_curve.png`
- `artifacts_p2/p2_confusion_matrix.png`
- `yproba1_test.txt` (rename to `yproba2_test.txt` before submitting for Problem 2)

## Customize for your report

- Edit parameter grids to include more values.
- Add TF‑IDF (TfidfVectorizer) or character‑level n‑grams to Problem 1.
- In Problem 2, try fusing numeric features from `x_train.csv` with embeddings (feature concatenation) and compare.

## Repro and seeds

We set random_state where possible for reproducibility of CV splits and results.
