"""Task 1.3: fixed EMA features + training-only StandardScaler + logistic regression."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from data import load_har, subject_split, subset_indices
from signal_utils import ema_features
from train import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/har")
    p.add_argument("--download", action="store_true")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1,0.5,0.9,0.99])
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--subset_seed", type=int, default=123)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--val_subjects", type=int, nargs="+")
    p.add_argument("--train_subset", type=int)
    p.add_argument("--out", default="runs/ema")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    x, y, subjects = load_har(args.data_root, download=args.download)
    tr, va = subject_split(subjects, args.val_fraction, args.split_seed, args.val_subjects)
    tr = tr[subset_indices(y[tr], args.train_subset, args.subset_seed)]
    xtr, xva = ema_features(x[tr], args.alphas), ema_features(x[va], args.alphas)
    out = Path(args.out)
    if (out / "summary.json").exists():
        raise FileExistsError("choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        pipeline.fit(xtr, y[tr])
        result = {"seed": seed, **metrics(y[va], pipeline.predict(xva), 6)}
        results.append(result)
        joblib.dump({"pipeline": pipeline, "alphas": args.alphas,
                     "val_subjects": np.unique(subjects[va]).tolist()}, out / f"seed_{seed}.joblib")
    summary = {"split": "validation", "runs": results, "alphas": args.alphas, "config": vars(args),
               "train_subjects": np.unique(subjects[tr]).tolist(), "val_subjects": np.unique(subjects[va]).tolist(),
               "feature_dim": xtr.shape[1],
               "note": "lbfgs logistic regression is deterministic; seed variance may be exactly zero."}
    for key in ["accuracy", "macro_f1"]:
        v = [r[key] for r in results]
        summary[key] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v)>1 else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
