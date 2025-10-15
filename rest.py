
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
