import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

from train_valid_test_loader import load_train_valid_test_datasets  # :contentReference[oaicite:1]{index=1}

# ============================================================
# PATHS
# ============================================================
DATA_DIR = "data_movie_lens_100k"
DEV_CSV = os.path.join(DATA_DIR, "ratings_all_development_set.csv")
LEADERBOARD_CSV = os.path.join(DATA_DIR, "ratings_masked_leaderboard_set.csv")

OUT_DIR = "prob2_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD TRAIN/VALID/TEST (same split from Problem 1)
# ============================================================
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# Build Surprise-friendly dataframe using train+valid
train_user, train_item, train_rating = train_tuple
valid_user, valid_item, valid_rating = valid_tuple

df_train_valid = pd.DataFrame({
    "user": np.concatenate([train_user, valid_user]),
    "item": np.concatenate([train_item, valid_item]),
    "rating": np.concatenate([train_rating, valid_rating]),
})

reader = Reader(rating_scale=(1, 5))
data_train_valid = Dataset.load_from_df(df_train_valid[["user", "item", "rating"]], reader)


# ============================================================
# 2. GRID SEARCH FOR BEST SVD (Problem 2b)
# ============================================================
param_grid = {
    "n_factors": [20, 50, 80, 120],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.02, 0.05]
}

print("Running SVD GridSearch...")
gs = GridSearchCV(
    algo_class=SVD,
    param_grid=param_grid,
    measures=["mae"],
    cv=3,
    n_jobs=-1,
    refit=True
)
gs.fit(data_train_valid)

best_params = gs.best_params["mae"]
best_cv_mae = gs.best_score["mae"]

print("Best parameters:", best_params)
print("Best CV MAE:", best_cv_mae)

# Save raw results for plotting
results = pd.DataFrame(gs.cv_results)
results.to_csv(os.path.join(OUT_DIR, "svd_grid_results.csv"), index=False)


# ============================================================
# 3. TRAIN FINAL SVD MODEL ON TRAIN+VALID
# ============================================================
print("\nTraining final SVD model...")
algo = SVD(
    n_factors=best_params["n_factors"],
    lr_all=best_params["lr_all"],
    reg_all=best_params["reg_all"],
    random_state=1376
)

trainset_full = data_train_valid.build_full_trainset()
algo.fit(trainset_full)


# ============================================================
# 4. EVALUATE ON DEV TEST SPLIT (Problem 2c)
# ============================================================
print("\nEvaluating on test split...")

test_user, test_item, test_rating = test_tuple
abs_errors = []

for u, i, r in zip(test_user, test_item, test_rating):
    pred_obj = algo.predict(uid=int(u), iid=int(i))
    abs_errors.append(abs(pred_obj.est - float(r)))

dev_test_mae = float(np.mean(abs_errors))

print("Dev Test MAE:", dev_test_mae)
with open(os.path.join(OUT_DIR, "dev_test_mae.txt"), "w") as f:
    f.write(f"{dev_test_mae:.6f}\n")


# ============================================================
# 5. GENERATE LEADERBOARD PREDICTIONS (10000 lines)
# ============================================================
df_lb = pd.read_csv(LEADERBOARD_CSV)

preds = []
print("\nGenerating leaderboard predictions...")
for _, row in df_lb.iterrows():
    uid = int(row["user_id"])
    iid = int(row["item_id"])
    preds.append(algo.predict(uid, iid).est)

preds = np.array(preds)
np.savetxt("predicted_ratings_leaderboard.txt", preds, fmt="%.6f")

print("Saved: predicted_ratings_leaderboard.txt")


# ============================================================
# 6. PLOT HYPERPARAMETER SELECTION (Problem 2b Figure)
# ============================================================
grouped = results.groupby("param_n_factors")["mean_test_mae"].min().reset_index()

plt.figure(figsize=(7, 5))
plt.plot(grouped["param_n_factors"], grouped["mean_test_mae"], marker="o")
plt.xlabel("n_factors")
plt.ylabel("Cross-validated MAE")
plt.title("Problem 2b: SVD Hyperparameter Selection Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "problem2b_svd_curve.png"))
plt.close()

print("\nGenerated outputs:")
print(" - dev_test_mae.txt")
print(" - svd_grid_results.csv")
print(" - problem2b_svd_curve.png")
print(" - predicted_ratings_leaderboard.txt")
