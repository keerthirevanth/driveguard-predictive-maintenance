"""Milestone 2 - LightGBM point-in-time baseline with honest time-split evaluation.

Establishes the bar the Milestone 3 bake-off must beat. Trained on the downsampled
train split; evaluated on the NATURAL-distribution val and test splits (batched scoring
so the full test quarter never loads at once).

Categorical `model` is mapped to integer codes learned on train (unseen -> -1) and passed
to LightGBM as a categorical feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from driveguard.evaluation.metrics import evaluate_classification

CAT_COL = "model"


def _model_code_map(train_path: Path) -> dict[str, int]:
    models = pl.read_parquet(train_path, columns=[CAT_COL])[CAT_COL].unique().to_list()
    return {m: i for i, m in enumerate(sorted(x for x in models if x is not None))}


def _to_xy(df: pl.DataFrame, feature_cols: list[str], code_map: dict[str, int]):
    df = df.with_columns(
        pl.col(CAT_COL).replace_strict(code_map, default=-1).cast(pl.Int32).alias(CAT_COL)
    )
    X = df.select(feature_cols).to_numpy()
    y = df["y"].to_numpy().astype(np.int8)
    return X, y


def _score_parquet(model, path: Path, feature_cols: list[str], code_map: dict[str, int],
                   batch_rows: int = 2_000_000):
    """Batched scoring of a (possibly huge) split. Returns (y_true, y_score)."""
    ys, ss = [], []
    pf = pq.ParquetFile(path)
    cols = [*feature_cols, "y"] if CAT_COL in feature_cols else [*feature_cols, "y", CAT_COL]
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = pl.from_arrow(batch)
        X, y = _to_xy(df, feature_cols, code_map)
        ss.append(model.predict_proba(X)[:, 1].astype(np.float32))
        ys.append(y)
    return np.concatenate(ys), np.concatenate(ss)


def run(cfg: dict, project_root: Path, feature_set: str = "big5", horizon: int = 30) -> dict:
    data_dir = project_root / cfg["data"]["processed_dir"] / f"features_{feature_set}_N{horizon}"
    cat_idx_name = [CAT_COL, "capacity_gb", "drive_age_days"]
    smart = [c for c in pl.read_parquet(data_dir / "train.parquet").columns
             if c.endswith("_raw")]
    feature_cols = cat_idx_name + smart

    code_map = _model_code_map(data_dir / "train.parquet")
    train = pl.read_parquet(data_dir / "train.parquet")
    Xtr, ytr = _to_xy(train, feature_cols, code_map)
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())

    clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=neg / pos, n_jobs=-1, random_state=42, verbose=-1,
    )
    clf.fit(Xtr, ytr, categorical_feature=[0])  # model is column 0

    results = {"feature_set": feature_set, "horizon": horizon,
               "train_rows": len(ytr), "train_pos": pos, "train_neg": neg,
               "feature_cols": feature_cols}
    for split in ["val", "test"]:
        yt, ys = _score_parquet(clf, data_dir / f"{split}.parquet", feature_cols, code_map)
        results[split] = evaluate_classification(yt, ys, cfg)

    imp = dict(sorted(zip(feature_cols, clf.feature_importances_.tolist()),
                      key=lambda kv: -kv[1]))
    results["feature_importance"] = imp

    out_dir = project_root / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"baseline_{feature_set}_N{horizon}.json").write_text(json.dumps(results, indent=2))
    model_dir = project_root / "models_store"
    model_dir.mkdir(exist_ok=True)
    clf.booster_.save_model(str(model_dir / f"baseline_{feature_set}_N{horizon}.txt"))
    return results


if __name__ == "__main__":
    import sys

    from driveguard.config import PROJECT_ROOT, load_config

    cfg = load_config()
    fs = sys.argv[1] if len(sys.argv) > 1 else "big5"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    res = run(cfg, PROJECT_ROOT, fs, n)
    print(json.dumps({k: v for k, v in res.items() if k not in ("feature_importance",)}, indent=2))
    print("\nfeature_importance:", json.dumps(res["feature_importance"]))
