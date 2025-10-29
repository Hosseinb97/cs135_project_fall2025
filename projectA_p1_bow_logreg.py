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
    stop_words=None,    # drop stopwords
    min_df=2,                # ignore rare words (appearing in <5 docs)
    max_df=0.9,              # ignore overly common words (>60% docs)
    binary=False,             # use counts, not just presence
    ngram_range=(1,2)
)

# Get features and labels
X_texts = x_train_df["text"].values
y = y_train_df["y_bin"].values

# Fit and transform
X_bow = vectorizer.fit_transform(X_texts)

vocab_size = len(vectorizer.vocabulary_)
print(f"Vocabulary size: {vocab_size}")

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
    verbose=2,
    return_train_score=True,
)

for train_idx, val_idx in cv.split(X_bow, y):
    print(len(train_idx), len(val_idx))

grid.fit(X_bow, y)

print("\n=== Best hyperparameters ===")
print(grid.best_params_)
print(f"Best mean AUROC across folds: {grid.best_score_:.4f}")

# -----------------------------
# Plot AUROC vs C for both penalties
# -----------------------------
results = pd.DataFrame(grid.cv_results_)

plt.figure(figsize=(8, 6))

for penalty in results["param_penalty"].unique():
    subset = results[results["param_penalty"] == penalty]
    C_vals = np.array(subset["param_C"])

    val_fold_scores = np.stack([
        subset[f"split{i}_test_score"].values for i in range(N_FOLDS)
    ])
    val_mean = val_fold_scores.mean(axis=0)
    val_std = val_fold_scores.std(axis=0)

    plt.plot(C_vals, subset["mean_train_score"], "o--", label=f"Train ({penalty})")
    plt.plot(C_vals, val_mean, "o-", label=f"Validation ({penalty})")

    plt.fill_between(C_vals, val_mean - val_std, val_mean + val_std, alpha=0.15)
    for i in range(N_FOLDS):
        plt.scatter(C_vals, val_fold_scores[i, :], color="gray", alpha=0.4, s=15)

plt.xscale("log")
plt.xlabel("Regularization Strength (C)")
plt.ylabel("Held-out AUROC")
plt.title("Training vs Validation Performance Across Model Complexity")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()

plt.figtext(0.5, -0.05,
            "Figure: AUROC on training and validation sets as a function of regularization strength (C). "
            "Each point shows average performance across 5 folds, with shaded regions indicating ±1 SD. "
            "Low C values (<0.0.1) show underfitting (both scores low), while high C values (>10) show overfitting (train high, validation drops).",
            wrap=True, ha='center', fontsize=9)

plt.savefig("p1_hyperparam_train_val_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure → p1_hyperparam_train_val_curve.pdf")






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





# --- Identify and inspect validation predictions ---
y_val_true = y[val_idx]
y_val_pred = y_val_pred  # from earlier
y_val_proba = best_model.predict_proba(X_bow[val_idx])[:, 1]

# Attach metadata for inspection
val_df = y_train_df.iloc[val_idx].copy()
val_df["predicted"] = y_val_pred
val_df["true"] = y_val_true
val_df["prob_ks4_5"] = y_val_proba
val_df["text"] = x_train_df.iloc[val_idx]["text"].values

# Misclassifications
misclassified = val_df[val_df["predicted"] != val_df["true"]]
print(f"Total misclassified: {len(misclassified)} / {len(val_df)}")

# Look at the first few
for i, row in misclassified.head(5).iterrows():
    print("="*80)
    print(f"Title: {row['title']}  |  Author: {row['author']}")
    print(f"True label: {row['Coarse Label']}  |  Predicted: "
          f"{'Key Stage 4-5' if row['predicted']==1 else 'Key Stage 2-3'}")
    print(f"Predicted probability (KS4-5): {row['prob_ks4_5']:.3f}")
    print("Excerpt:")
    print(textwrap.fill(row['text'], width=100))
    print()

val_df["length"] = val_df["text"].str.split().apply(len)
print("Average length by true label:")
print(val_df.groupby("true")["length"].mean())

print("\nAverage length of misclassified vs correct:")
print(val_df.assign(correct = val_df["true"] == val_df["predicted"])
      .groupby("correct")["length"].mean())

print("\nAuthors with most misclassifications:")
print(misclassified["author"].value_counts().head())









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

