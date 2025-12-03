import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

# -----------------------------
# Utility functions
# -----------------------------
def compute_perf(model, data_tuple):
    user, item, rating = data_tuple
    perf = model.evaluate_perf_metrics(user, item, rating)
    return perf["rmse"], perf["mae"]

def train_and_trace(K, alpha, step_size, n_epochs, train_tuple, valid_tuple):
    model = CollabFilterOneVectorPerItem(
        n_epochs=n_epochs,
        batch_size=1000,
        step_size=step_size,
        n_factors=K,
        alpha=alpha
    )
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)

    return model, model.trace_epoch, model.trace_rmse_train, model.trace_rmse_valid


# -----------------------------
# Load dataset
# -----------------------------
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# -----------------------------
# 1a — RMSE vs epoch for alpha=0
# -----------------------------
Ks = [2, 10, 50]
step_size = 500.0
n_epochs = 400

results_1a = []

plt.figure(figsize=(18, 4))

for i, K in enumerate(Ks):
    model, epoch_trace, train_rmse, valid_rmse = train_and_trace(
        K=K,
        alpha=0.0,
        step_size=step_size,
        n_epochs=n_epochs,
        train_tuple=train_tuple,
        valid_tuple=valid_tuple
    )
    results_1a.append((K, model))

    plt.subplot(1, 3, i+1)
    plt.plot(epoch_trace, train_rmse, label="Train RMSE")
    plt.plot(epoch_trace, valid_rmse, label="Valid RMSE")
    plt.title(f"K={K}, alpha=0")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()

plt.suptitle("1a: RMSE vs Epoch (K=2,10,50; alpha=0)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("1a_rmse_traces.png")
plt.close()


# -----------------------------
# 1b — K=50, alpha > 0
# -----------------------------
alpha_list = [1e-5, 1e-4, 2e-4, 1e-3]
results_1b = []

best_alpha = None
best_valid_rmse = np.inf
best_model = None
best_traces = None

for alpha in alpha_list:
    model, epoch_trace, train_rmse, valid_rmse = train_and_trace(
        K=50,
        alpha=alpha,
        step_size=step_size,
        n_epochs=n_epochs,
        train_tuple=train_tuple,
        valid_tuple=valid_tuple,
    )

    # Pick best alpha based on last validation RMSE
    if valid_rmse[-1] < best_valid_rmse:
        best_valid_rmse = valid_rmse[-1]
        best_alpha = alpha
        best_model = model
        best_traces = (epoch_trace, train_rmse, valid_rmse)

# Plot best alpha
epoch_trace, train_rmse, valid_rmse = best_traces
plt.figure(figsize=(6, 4))
plt.plot(epoch_trace, train_rmse, label="Train RMSE")
plt.plot(epoch_trace, valid_rmse, label="Valid RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"1b: K=50, Best alpha={best_alpha}")
plt.legend()
plt.tight_layout()
plt.savefig("1b_k50_alpha_trace.png")
plt.close()


# -----------------------------
# 1c — Table of RMSE & MAE
# -----------------------------
rows = []

# For each K in {2,10,50}, alpha=0
for K, model in results_1a:
    tr_rmse, tr_mae = compute_perf(model, train_tuple)
    va_rmse, va_mae = compute_perf(model, valid_tuple)
    te_rmse, te_mae = compute_perf(model, test_tuple)

    rows.append({
        "K": K,
        "alpha": 0.0,
        "train_rmse": tr_rmse, "train_mae": tr_mae,
        "valid_rmse": va_rmse, "valid_mae": va_mae,
        "test_rmse": te_rmse,  "test_mae": te_mae,
    })

# Add best alpha for K=50
tr_rmse, tr_mae = compute_perf(best_model, train_tuple)
va_rmse, va_mae = compute_perf(best_model, valid_tuple)
te_rmse, te_mae = compute_perf(best_model, test_tuple)

rows.append({
    "K": 50,
    "alpha": best_alpha,
    "train_rmse": tr_rmse, "train_mae": tr_mae,
    "valid_rmse": va_rmse, "valid_mae": va_mae,
    "test_rmse": te_rmse,  "test_mae": te_mae,
})

df = pd.DataFrame(rows)
df.to_csv("1c_results_table.csv", index=False)
print(df)


# -----------------------------
# 1d — Embedding scatter for selected movies
# -----------------------------
import csv

# Load selected movies
select_movies = pd.read_csv("select_movies.csv")

# Use V matrix from best model (K=50, alpha=best)
V = best_model.param_dict["V"]

# If K > 2 → project to 2D via PCA/SVD
if V.shape[1] > 2:
    V_center = V - V.mean(axis=0)
    U_svd, S_svd, VT_svd = np.linalg.svd(V_center, full_matrices=False)
    V_2d = V_center.dot(VT_svd.T[:, :2])
else:
    V_2d = V[:, :2]

plt.figure(figsize=(8, 6))

for idx, row in select_movies.iterrows():
    movie_id = int(row["item_Id"])
    title = row["title"]
    x, y = V_2d[movie_id]
    plt.scatter(x, y)
    plt.text(x + 0.003, y + 0.003, title, fontsize=9)

plt.title("1d: 2D Movie Embedding (K=50, best alpha)")
plt.tight_layout()
plt.savefig("1d_embedding_scatter.png")
plt.close()

print("\nGenerated:")
print("- 1a_rmse_traces.png")
print("- 1b_k50_alpha_trace.png")
print("- 1c_results_table.csv")
print("- 1d_embedding_scatter.png")
