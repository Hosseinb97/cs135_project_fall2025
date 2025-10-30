# ==============================================================
# === Problem 1: Bag-of-Words + Logistic Regression ============
# ==============================================================

### Importations ###

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import textwrap

# --- force script to run in its own folder ---
#os.chdir(r"C:\Users\hbazda01\OneDrive - Tufts\CEE PhD @TUFTS\Fall 2025\intro to ml-cs135\Proj A") ###!!!! note 4 AJ: CHANGE THIS TO YOUR OWN PATH !!!!###


#read data using code from given snippet
if __name__ == '__main__':
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))
    print("Shape of x_test_df: %s" % str(x_test_df.shape))

    # Print out 8 random entries
    tr_text_list = x_train_df['text'].values.tolist()
    prng = np.random.RandomState(101)
    rows = prng.permutation(np.arange(y_train_df.shape[0]))
    for row_id in rows[:8]:
        text = tr_text_list[row_id]
        print("row %5d | %s BY %s | y = %s" % (
            row_id,
            y_train_df['title'].values[row_id],
            y_train_df['author'].values[row_id],
            y_train_df['Coarse Label'].values[row_id],
            ))
        # Pretty print text via textwrap library
        line_list = textwrap.wrap(tr_text_list[row_id],
            width=70,
            initial_indent='  ',
            subsequent_indent='  ')
        print('\n'.join(line_list))
        print("")

# -----------------------------
# CONFIG
# -----------------------------
N_FOLDS = 5
RANDOM_STATE = 101
SCORING = "roc_auc"

# -----------------------------
# the PIPELINE
# -----------------------------
if __name__ == '__main__':
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df  = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    # Map text labels to binary 0/1
    label_map = {
        "Key Stage 2-3": 0,
        "Key Stage 4-5": 1,
    }
    y_train_df = y_train_df.copy()
    y_train_df["y_bin"] = y_train_df["Coarse Label"].map(label_map)
    assert not y_train_df["y_bin"].isna().any(), "Unexpected label values"
    print(y_train_df["y_bin"].value_counts())

# -----------------------------
# Define a simple BoW vectorizer
vectorizer = CountVectorizer(
    lowercase=True,          # convert to lowercase
    stop_words="english",    # drop stopwords
    min_df=5,                # ignore rare words (appearing in <5 docs)
    max_df=0.6,              # ignore overly common words (>60% docs)
    binary=False             # use counts, not just presence
)

# Get features and labels
X_texts = x_train_df["text"].values
y = y_train_df["y_bin"].values

# Fit and transform
X_bow = vectorizer.fit_transform(X_texts)
print(f"BoW feature matrix shape: {X_bow.shape}")

# -----------------------------
# Define Logistic Regression model
# -----------------------------
log_reg = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",  # helps if class imbalance exists
    random_state=RANDOM_STATE
)





####Partner 2#############################
# -----------------------------
# Cross-Validation + Grid Search
# -----------------------------
param_grid = {
    "C": np.logspace(-3, 3, 7),  # Regularization strengths
    "penalty": ["l1", "l2"]
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring=SCORING,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_bow, y)

print("\n=== Best hyperparameters ===")
print(grid.best_params_)
print(f"Best mean AUROC across folds: {grid.best_score_:.4f}")

# -----------------------------
# Plot AUROC vs C for both penalties
# -----------------------------
results = pd.DataFrame(grid.cv_results_)
plt.figure(figsize=(7,5))
for penalty in results["param_penalty"].unique():
    subset = results[results["param_penalty"] == penalty]
    plt.semilogx(subset["param_C"], subset["mean_test_score"], marker='o', label=f"{penalty}")
plt.xlabel("Regularization Strength (C)")
plt.ylabel("Mean CV AUROC")
plt.title("Hyperparameter Search: Logistic Regression (BoW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("p1_hyperparam_search.png", dpi=300)
plt.close()

# -----------------------------
# Confusion Matrix on held-out fold (simple check)
# -----------------------------
best_model = grid.best_estimator_

# Use 1 CV split for visualization
train_idx, val_idx = next(cv.split(X_bow, y))
best_model.fit(X_bow[train_idx], y[train_idx])
y_val_pred = best_model.predict(X_bow[val_idx])

cm = confusion_matrix(y[val_idx], y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["KS2-3","KS4-5"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Validation Fold")
plt.tight_layout()
plt.savefig("p1_confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# Final Model Training + Test Prediction
# -----------------------------
# Train on full data with best params
final_model = LogisticRegression(
    solver="liblinear",
    penalty=grid.best_params_["penalty"],
    C=grid.best_params_["C"],
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE
)
final_model.fit(X_bow, y)

# Transform test data
X_test_bow = vectorizer.transform(x_test_df["text"].values)
y_test_proba = final_model.predict_proba(X_test_bow)[:,1]

# Save predictions
np.savetxt("yproba1_test.txt", y_test_proba, fmt="%.6f")
print("Saved test-set probabilities → yproba1_test.txt")
