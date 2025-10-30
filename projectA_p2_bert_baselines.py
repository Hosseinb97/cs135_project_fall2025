"""
- Loads x_train.csv, y_train.csv, x_test.csv and precomputed BERT embeddings:
    x_train_BERT_embeddings.npz, x_test_BERT_embeddings.npz (H=768)
- Runs StratifiedKFold CV (optionally with groups by author) to select hyperparams for a chosen classifier
- Supported classifiers: logistic (default), linear_svm, mlp
- Saves to out_dir:
    - Hyperparameter curve for a key complexity parameter
    - Confusion matrix for best fold
    - Best params JSON
    - yproba1_test.txt for leaderboard (NOTE: For Problem 2, rename to yproba2_test.txt before submission)
Important:
- You MUST rename the output file to yproba2_test.txt when submitting to the Problem 2 leaderboard.
- This is a baseline; feel free to extend with TF-IDF, numeric features, late fusion, etc.
"""
import argparse, os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV

def load_arr_from_npz(npz_path):
    f = np.load(npz_path)
    arr = f.f.arr_0.copy()
    f.close()
    return arr

def load_xyH(data_dir):
    x_train = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    x_test  = pd.read_csv(os.path.join(data_dir, "x_test.csv"))
    Xtr = load_arr_from_npz(os.path.join(data_dir, "x_train_BERT_embeddings.npz"))
    Xte = load_arr_from_npz(os.path.join(data_dir, "x_test_BERT_embeddings.npz"))
    y = y_train["Coarse Label"].map({"Key Stage 2-3":0, "Key Stage 4-5":1}).astype(int).values
    groups = x_train["author"].astype(str).values
    return Xtr, y, groups, Xte, x_train, y_train, x_test

def get_estimator_and_grid(name):
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=1000, solver="liblinear", class_weight='balanced')
        grid = {"C":np.logspace(-3,2,10), "penalty":["l2","l1"]}
        key = "C"
    elif name == "linear_svm":
        base = LinearSVC(dual=False)
        est = CalibratedClassifierCV(base, cv=5)
        grid = {"base_estimator__C":np.logspace(-3,2,10)}
        key = "base_estimator__C"
    elif name == "mlp":
        est = MLPClassifier(max_iter=2000, early_stopping=True, activation='relu')
        grid = {"hidden_layer_sizes":[(128,), (256,), (128,64), (256,128)], "alpha":[1e-5, 1e-4, 1e-3],
                "learning_rate_init":[1e-3, 1e-4]}
        key = "hidden_layer_sizes"
    else:
        raise ValueError("Unknown clf name")
    return est, grid, key

def plot_curve(results_df, key, out_png, title):
    xs = sorted(results_df[f"param_{key}"].unique(), key=lambda z: (tuple(z) if isinstance(z, tuple) else z))
    val_means, val_stds, tr_means = [], [], []
    for x in xs:
        sub = results_df[results_df[f"param_{key}"]==x]
        best = sub.sort_values("mean_test_score", ascending=False).iloc[0]
        val_means.append(best["mean_test_score"])
        val_stds.append(best["std_test_score"])
        tr_means.append(best["mean_train_score"])
    plt.figure()
    plt.errorbar(range(len(xs)), val_means, yerr=val_stds, marker="o", label="Validation AUROC")
    plt.plot(range(len(xs)), tr_means, marker="s", label="Training AUROC")
    plt.xticks(range(len(xs)), [str(x) for x in xs], rotation=30, ha="right")
    plt.xlabel(key)
    plt.ylabel("AUROC")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def make_confusion(best_estimator, X, y, groups, out_png):
    try:
        cv_vis = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
        splits = list(cv_vis.split(np.zeros_like(y), y, groups))
    except:
        cv_vis = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        splits = list(cv_vis.split(X, y))
    tr, va = splits[0]
    best_estimator.fit(X[tr], y[tr])
    y_proba = best_estimator.predict_proba(X[va])[:,1]
    y_hat = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y[va], y_hat, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Lower (0)","Upper (1)"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Problem 2: Confusion Matrix (one CV fold, thr=0.5)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./artifacts_p2")
    ap.add_argument("--clf", type=str, default="logistic", choices=["logistic","linear_svm","mlp"])
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Xtr, y, groups, Xte, x_train_df, y_train_df, x_test_df = load_xyH(args.data_dir)
    est, grid, key = get_estimator_and_grid(args.clf)

    try:
        cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(est, grid, scoring="roc_auc", n_jobs=-1, cv=cv, refit=True, return_train_score=True)
        gs.fit(Xtr, y, groups=groups)
    except Exception as e:
        print("[WARN] Using StratifiedKFold:", e)
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(est, grid, scoring="roc_auc", n_jobs=-1, cv=cv, refit=True, return_train_score=True)
        gs.fit(Xtr, y)

    print("Best params:", gs.best_params_)
    print("Best CV AUROC: %.4f" % gs.best_score_)
    with open(os.path.join(args.out_dir, "p2_best_params.json"), "w") as f:
        json.dump({"best_params": gs.best_params_, "best_cv_auroc": float(gs.best_score_), "clf": args.clf}, f, indent=2)

    results_df = pd.DataFrame(gs.cv_results_)
    plot_curve(results_df, key, os.path.join(args.out_dir, "p2_hyperparam_curve.png"),
               f"Problem 2: Hyperparameter Curve ({args.clf})")

    make_confusion(gs.best_estimator_, Xtr, y, groups, os.path.join(args.out_dir, "p2_confusion_matrix.png"))

    best = gs.best_estimator_
    best.fit(Xtr, y)
    yproba_test = best.predict_proba(Xte)[:,1]
    # Save provisional name; rename to yproba2_test.txt yourself for submission
    np.savetxt("yproba2_test.txt", yproba_test, fmt="%.6f")
    print("Wrote provisional probabilities to yproba2_test.txt. Rename to yproba2_test.txt before submitting Problem 2.")

if __name__ == "__main__":
    main()
