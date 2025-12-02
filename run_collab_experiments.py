# run_collab_experiments.py
# Requires:
#   - autograd
#   - numpy
#   - pandas
#   - matplotlib
#   - your modules:
#       AbstractBaseCollabFilterSGD.py
#       CollabFilterOneVectorPerItem.py
#       train_valid_test_loader.py
#
# Place this script in the same directory as those modules and run:
#   python run_collab_experiments.py

import os
import copy
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RANDOM_STATE = 20190415

from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

BATCH_SIZE = 1000

STEP_SIZE = 0.5
N_EPOCHS = 80
PATIENCE = 8
TRACE_REPORT_FREQ = None

OUTPUT_DIR = "collab_experiment_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(model, param_dict, data_tuple):
    """Helper to compute rmse/mae on a dataset using model.param_dict temporarily"""
    user_ids, item_ids, ratings = data_tuple
    yhat = model.predict(user_ids, item_ids, **param_dict)
    yhat = np.array(yhat)
    ratings = np.array(ratings)
    rmse = math.sqrt(np.mean((yhat - ratings) ** 2))
    mae = np.mean(np.abs(yhat - ratings))
    return {"rmse": rmse, "mae": mae}


def early_stopping_trace(trace_rmse_valid, patience):
    """Return True if should stop (no improvement in last `patience` epochs)."""
    if len(trace_rmse_valid) <= patience:
        return False
    best_so_far = min(trace_rmse_valid[:-patience])
    recent_best = min(trace_rmse_valid[-patience:])
    return recent_best >= best_so_far


def train_with_early_stopping(model,
                              train_tuple,
                              valid_tuple,
                              test_tuple,
                              step_size,
                              n_epochs,
                              patience,
                              do_warm_start=False):
    """Train model, monitor validation RMSE, perform early stopping.
    Returns dict with training artifacts.
    """
    model.step_size = step_size
    model.n_epochs = n_epochs
    model.batch_size = BATCH_SIZE

    all_user_ids = np.concatenate([train_tuple[0], valid_tuple[0], test_tuple[0]])
    all_item_ids = np.concatenate([train_tuple[1], valid_tuple[1], test_tuple[1]])
    n_users = int(all_user_ids.max() + 1)
    n_items = int(all_item_ids.max() + 1)

    model.init_parameter_dict(n_users, n_items, train_tuple)

    best_val_rmse = float("inf")
    best_snapshot = None
    best_epoch = None


    total_epochs_run = 0
    wait = 0

    for epoch_block in range(n_epochs):
        model.n_epochs = 1
        model.fit(train_tuple, valid_tuple, do_warm_start=True if total_epochs_run > 0 else False)

        total_epochs_run += 1

        perf_valid = compute_metrics(model, model.param_dict, valid_tuple)
        cur_val_rmse = perf_valid["rmse"]

        if cur_val_rmse < best_val_rmse - 1e-12:
            best_val_rmse = cur_val_rmse
            best_snapshot = dict(
                epoch=total_epochs_run,
                param_dict=copy.deepcopy({k: np.array(v) for k, v in model.param_dict.items()}),
                trace_epoch=copy.deepcopy(model.trace_epoch),
                trace_rmse_train=copy.deepcopy(model.trace_rmse_train),
                trace_rmse_valid=copy.deepcopy(model.trace_rmse_valid),
                trace_mae_train=copy.deepcopy(model.trace_mae_train),
                trace_mae_valid=copy.deepcopy(model.trace_mae_valid),
                trace_loss=copy.deepcopy(model.trace_loss),
            )
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {total_epochs_run} (no val improvement for {patience} epochs).")
            break

    final_snapshot = dict(
        total_epochs_run=total_epochs_run,
        best_snapshot=best_snapshot,
        final_param_dict=copy.deepcopy({k: np.array(v) for k, v in model.param_dict.items()}),
        final_trace_epoch=copy.deepcopy(model.trace_epoch),
        final_trace_rmse_train=copy.deepcopy(model.trace_rmse_train),
        final_trace_rmse_valid=copy.deepcopy(model.trace_rmse_valid),
        final_trace_mae_train=copy.deepcopy(model.trace_mae_train),
        final_trace_mae_valid=copy.deepcopy(model.trace_mae_valid),
        final_trace_loss=copy.deepcopy(model.trace_loss),
    )
    return final_snapshot


def run_step3i(train_tuple, valid_tuple, test_tuple, n_factors_list,
                step_size, n_epochs, patience, random_state):
    """Run Step 3(i): n_factors in n_factors_list, alpha=0"""
    results = []
    for k in n_factors_list:
        print(f"\n=== Training K={k}, alpha=0 ===")
        model = CollabFilterOneVectorPerItem(
            step_size=step_size, n_epochs=1,
            batch_size=BATCH_SIZE, n_factors=k, alpha=0.0,
            random_state=random_state)
        train_artifacts = train_with_early_stopping(
            model, train_tuple, valid_tuple, test_tuple,
            step_size=step_size, n_epochs=n_epochs, patience=patience)
        best = train_artifacts["best_snapshot"]
        if best is None:
            best_param_dict = train_artifacts["final_param_dict"]
            best_epoch = train_artifacts["total_epochs_run"]
        else:
            best_param_dict = best["param_dict"]
            best_epoch = best["epoch"]

        train_perf = compute_metrics(model, best_param_dict, train_tuple)
        valid_perf = compute_metrics(model, best_param_dict, valid_tuple)
        test_perf = compute_metrics(model, best_param_dict, test_tuple)

        res = {
            "K": k,
            "alpha": 0.0,
            "best_epoch": best_epoch,
            "train_perf": train_perf,
            "valid_perf": valid_perf,
            "test_perf": test_perf,
            "param_dict": best_param_dict,
            "trace_epoch": train_artifacts["final_trace_epoch"],
            "trace_rmse_train": train_artifacts["final_trace_rmse_train"],
            "trace_rmse_valid": train_artifacts["final_trace_rmse_valid"],
            "trace_mae_train": train_artifacts["final_trace_mae_train"],
            "trace_mae_valid": train_artifacts["final_trace_mae_valid"],
        }
        results.append(res)
        out_base = os.path.join(OUTPUT_DIR, f"k{str(k)}_alpha0")
        with open(out_base + "_params.pkl", "wb") as f:
            pickle.dump(best_param_dict, f)
        pd.DataFrame({
            "metric": ["train_rmse", "train_mae", "valid_rmse", "valid_mae", "test_rmse", "test_mae"],
            "value": [train_perf["rmse"], train_perf["mae"],
                      valid_perf["rmse"], valid_perf["mae"],
                      test_perf["rmse"], test_perf["mae"]],
        }).to_csv(out_base + "_metrics.csv", index=False)
    return results


def run_step3ii(train_tuple, valid_tuple, test_tuple, k_target,
                alpha_list, step_size, n_epochs, patience, random_state):
    """Run Step 3(ii): n_factors=k_target with various alpha values"""
    results = []
    for alpha in alpha_list:
        print(f"\n=== Training K={k_target}, alpha={alpha} ===")
        model = CollabFilterOneVectorPerItem(
            step_size=step_size, n_epochs=1, batch_size=BATCH_SIZE,
            n_factors=k_target, alpha=alpha,
            random_state=random_state)
        train_artifacts = train_with_early_stopping(
            model, train_tuple, valid_tuple, test_tuple,
            step_size=step_size, n_epochs=n_epochs, patience=patience)
        best = train_artifacts["best_snapshot"]
        if best is None:
            best_param_dict = train_artifacts["final_param_dict"]
            best_epoch = train_artifacts["total_epochs_run"]
        else:
            best_param_dict = best["param_dict"]
            best_epoch = best["epoch"]

        train_perf = compute_metrics(model, best_param_dict, train_tuple)
        valid_perf = compute_metrics(model, best_param_dict, valid_tuple)
        test_perf = compute_metrics(model, best_param_dict, test_tuple)

        res = {
            "K": k_target,
            "alpha": alpha,
            "best_epoch": best_epoch,
            "train_perf": train_perf,
            "valid_perf": valid_perf,
            "test_perf": test_perf,
            "param_dict": best_param_dict,
            "trace_epoch": train_artifacts["final_trace_epoch"],
            "trace_rmse_train": train_artifacts["final_trace_rmse_train"],
            "trace_rmse_valid": train_artifacts["final_trace_rmse_valid"],
        }
        results.append(res)
        out_base = os.path.join(OUTPUT_DIR, f"k{str(k_target)}_alpha{str(alpha).replace('.','p')}")
        with open(out_base + "_params.pkl", "wb") as f:
            pickle.dump(best_param_dict, f)
        pd.DataFrame({
            "metric": ["train_rmse", "train_mae", "valid_rmse", "valid_mae", "test_rmse", "test_mae"],
            "value": [train_perf["rmse"], train_perf["mae"],
                      valid_perf["rmse"], valid_perf["mae"],
                      test_perf["rmse"], test_perf["mae"]],
        }).to_csv(out_base + "_metrics.csv", index=False)
    return results


def plot_step3i_traces(results_list, filename):
    """Plot RMSE vs epoch for K=2,10,50 side-by-side"""
    npanels = len(results_list)
    fig, axes = plt.subplots(1, npanels, figsize=(5 * npanels, 4), sharey=True)
    if npanels == 1:
        axes = [axes]
    ymin = float("inf")
    ymax = float("-inf")
    for res in results_list:
        tr = res["trace_rmse_train"]
        vr = res["trace_rmse_valid"]
        if len(tr) > 0:
            ymin = min(ymin, min(tr))
            ymax = max(ymax, max(tr))
        if len(vr) > 0:
            ymin = min(ymin, min(vr))
            ymax = max(ymax, max(vr))
    if ymin == float("inf"):
        ymin, ymax = 0.0, 2.0
    margin = 0.02 * (ymax - ymin if ymax > ymin else 1.0)
    for ax, res in zip(axes, results_list):
        ax.plot(res["trace_epoch"], res["trace_rmse_train"], label="train RMSE")
        ax.plot(res["trace_epoch"], res["trace_rmse_valid"], label="valid RMSE")
        ax.set_title(f"K={res['K']}, alpha={res['alpha']}")
        ax.set_xlabel("epoch")
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.legend()
    fig.suptitle("RMSE vs epoch for different K (consistent y-axis)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename)
    plt.close(fig)


def plot_alpha_traces(result_for_alphas, filename):
    """Plot RMSE vs epoch for K=target under different alphas (single panel).
    We'll plot the best-valid RMSE trace for the best alpha run."""
    # Choose alpha run with best final valid RMSE
    best = min(result_for_alphas, key=lambda r: r["valid_perf"]["rmse"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(best["trace_epoch"], best["trace_rmse_train"], label="train RMSE")
    ax.plot(best["trace_epoch"], best["trace_rmse_valid"], label="valid RMSE")
    ax.set_title(f"K={best['K']} best alpha={best['alpha']}")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return best


def produce_results_table(res_list_all, filename):
    """Produce a DataFrame summarizing train/valid/test RMSE & MAE for each row/run."""
    rows = []
    for r in res_list_all:
        rows.append({
            "K": r["K"],
            "alpha": r["alpha"],
            "best_epoch": r["best_epoch"],
            "train_rmse": r["train_perf"]["rmse"],
            "train_mae": r["train_perf"]["mae"],
            "valid_rmse": r["valid_perf"]["rmse"],
            "valid_mae": r["valid_perf"]["mae"],
            "test_rmse": r["test_perf"]["rmse"],
            "test_mae": r["test_perf"]["mae"],
        })
    df = pd.DataFrame(rows).sort_values(["K", "alpha"])
    df.to_csv(filename, index=False)
    return df


def embedding_scatter_for_select_movies(best_params, select_csv_path, out_png):
    """Plot 2D embedding scatter for select movies using V (item vectors).
       Expects select_csv_path to have movieId and title columns, and movieId aligned with dataset item ids.
       If movieId is not direct zero-based index, the user should provide mapping. This function will try
       to locate the movieId as an index into V; if not possible, it will try to match by title.
    """
    df_select = pd.read_csv(select_csv_path)
    V = best_params["V"]
    V = np.array(V)
    K = V.shape[1]
    if K > 2:
        V_center = V - V.mean(axis=0)
        U, S, VT = np.linalg.svd(V_center, full_matrices=False)
        comps = V_center.dot(VT.T[:, :2])
    else:
        comps = V
        if K == 1:
            comps = np.column_stack([V[:, 0], np.zeros(V.shape[0])])

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df_select.iterrows():
        mid = row.get("movieId", None)
        title = row.get("title", str(mid))
        point = None
        if mid is not None:
            try:
                idx = int(mid)
                if 0 <= idx < comps.shape[0]:
                    point = comps[idx]
            except Exception:
                point = None
        if point is None:
            print(f"Warning: could not locate movie '{title}' by id={mid}; skipping.")
            continue
        ax.scatter(point[0], point[1])
        ax.text(point[0] + 1e-6, point[1] + 1e-6, str(title), fontsize=9)
    ax.set_title("2D movie-embedding scatter for select movies")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
    train_tuple = tuple(np.array(x, dtype=int) if i < 2 else np.array(x, dtype=float)
                        for i, x in enumerate(train_tuple))
    valid_tuple = tuple(np.array(x, dtype=int) if i < 2 else np.array(x, dtype=float)
                        for i, x in enumerate(valid_tuple))
    test_tuple = tuple(np.array(x, dtype=int) if i < 2 else np.array(x, dtype=float)
                        for i, x in enumerate(test_tuple))

    k_list = [2, 10, 50]
    step_size_for_k = {2: 0.5, 10: 0.5, 50: 0.5}
    results_step3i = []
    for k in k_list:
        res_k = run_step3i(train_tuple, valid_tuple, test_tuple, [k],
                           step_size_for_k[k], N_EPOCHS, PATIENCE, DEFAULT_RANDOM_STATE)
        results_step3i.extend(res_k)

    plot_step3i_traces(results_step3i,
                       os.path.join(OUTPUT_DIR, "step3i_rmse_traces_k2_10_50.png"))

    alpha_candidates = [0.001, 0.01, 0.1, 1.0]
    results_step3ii = run_step3ii(train_tuple, valid_tuple, test_tuple,
                                  k_target=50, alpha_list=alpha_candidates,
                                  step_size=STEP_SIZE, n_epochs=N_EPOCHS, patience=PATIENCE,
                                  random_state=DEFAULT_RANDOM_STATE)
    best_alpha_run = plot_alpha_traces(results_step3ii,
                                       os.path.join(OUTPUT_DIR, "step3ii_best_alpha_trace.png"))

    all_results = results_step3i + results_step3ii
    df_table = produce_results_table(all_results, os.path.join(OUTPUT_DIR, "results_table.csv"))
    print("\nSummary table saved to:", os.path.join(OUTPUT_DIR, "results_table.csv"))
    print(df_table)

    best_overall = min(all_results, key=lambda r: r["valid_perf"]["rmse"])
    print("\nBest overall run (by valid RMSE):", best_overall["K"], best_overall["alpha"],
          "valid_rmse=", best_overall["valid_perf"]["rmse"])

    with open(os.path.join(OUTPUT_DIR, "best_overall_params.pkl"), "wb") as f:
        pickle.dump(best_overall["param_dict"], f)

    select_movies_csv = "select_movies.csv"
    if os.path.exists(select_movies_csv):
        embedding_scatter_for_select_movies(
            best_overall["param_dict"],
            select_movies_csv,
            os.path.join(OUTPUT_DIR, "embedding_select_movies.png"))
        print("Embedding scatter saved to embedding_select_movies.png")
    else:
        print("select_movies.csv not found in working directory; skipping embedding plot.")

    print("\nAll outputs saved to folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
