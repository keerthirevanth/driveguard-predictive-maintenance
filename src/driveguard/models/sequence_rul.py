"""Milestone 4 (part 2) - deep sequence RUL models: LSTM / GRU / 1D-CNN.

Each model reads a drive's last L days of raw SMART values and predicts remaining useful
life (days). Trained with a censoring-aware loss:
  - uncensored (drive failed):   SmoothL1(pred, true_rul)
  - censored (still alive):      penalise only if pred < observed_survival (relu(obs - pred))
so censored drives correctly push predictions to be "at least this long".

Risk = -predicted_RUL (shorter life => higher risk) feeds the concordance index; predicted
RUL feeds RUL MAE on the drives that actually failed. Same metrics as the classical models,
so results are directly comparable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from driveguard.evaluation.metrics import evaluate_survival


def _load(npz_path: Path):
    d = np.load(npz_path)
    return d["X"], d["event"], d["dur"]


def _make_model(kind: str, n_features: int):
    import torch.nn as nn

    class RNN(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.rnn = cell(n_features, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, 32, 3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):                 # x: [B, L, F] -> conv wants [B, F, L]
            z = self.net(x.transpose(1, 2)).squeeze(-1)
            return self.head(z).squeeze(-1)

    if kind == "lstm":
        return RNN(nn.LSTM)
    if kind == "gru":
        return RNN(nn.GRU)
    if kind == "cnn1d":
        return CNN()
    raise ValueError(kind)


def _censored_rul_loss(pred, dur, event):
    import torch
    import torch.nn.functional as F
    obs = F.smooth_l1_loss(pred[event], dur[event]) if event.any() else 0.0
    cens = ~event
    under = F.relu(dur[cens] - pred[cens]).mean() if cens.any() else 0.0
    return obs + cens


def _fit_eval(kind, Xtr, etr, dtr, Xte, ete, dte, rul_cap=200.0,
              epochs=25, batch=512, lr=1e-3, seed=42):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # standardise features using train stats
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ytr = np.clip(dtr, 0, rul_cap)  # train target capped for stability

    model = _make_model(kind, Xtr.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr),
                       torch.tensor(etr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for xb, db, eb in dl:
            xb, db, eb = xb.to(dev), db.to(dev), eb.to(dev).bool()
            opt.zero_grad()
            loss = _censored_rul_loss(model(xb), db, eb)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 8192):
            xb = torch.tensor(Xte[i:i + 8192]).to(dev)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds)
    return evaluate_survival(ete, dte, risk_score=-pred, pred_time=pred)


def run_sequence_rul(seq_dir: str | Path, models: list[str],
                     mlflow_uri: str | None = None, **kw) -> list[dict]:
    seq_dir = Path(seq_dir)
    Xtr, etr, dtr = _load(seq_dir / "train.npz")
    Xte, ete, dte = _load(seq_dir / "test.npz")

    try:
        import mlflow
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("driveguard_sequence_rul")
        have_mlflow = True
    except Exception:
        have_mlflow = False

    results = []
    for kind in models:
        t0 = time.time()
        try:
            metrics = _fit_eval(kind, Xtr, etr, dtr, Xte, ete, dte, **kw)
            res = {"model": kind, "train_rows": int(len(Xtr)),
                   "fit_sec": round(time.time() - t0, 1), "test": metrics, "status": "ok"}
        except Exception as e:
            res = {"model": kind, "status": "error", "error": str(e)}
        results.append(res)
        msg = (f"  c_index={res['test']['c_index']:.4f} rul_mae={res['test'].get('rul_mae_days'):.1f}"
               if res.get("status") == "ok" and res["test"].get("c_index") is not None else "")
        print(json.dumps({"model": kind, "status": res.get("status")}) + msg, flush=True)
        if have_mlflow and res.get("status") == "ok":
            with mlflow.start_run(run_name=f"seq_{kind}"):
                mlflow.log_param("model", kind)
                if res["test"].get("c_index") is not None:
                    mlflow.log_metric("test_c_index", res["test"]["c_index"])
                if res["test"].get("rul_mae_days") is not None:
                    mlflow.log_metric("test_rul_mae_days", res["test"]["rul_mae_days"])
    return results


if __name__ == "__main__":
    import sys

    from driveguard.config import PROJECT_ROOT

    sdir = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "data" / "processed" / "sequences")
    mdls = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lstm", "gru", "cnn1d"]
    board = run_sequence_rul(sdir, mdls, mlflow_uri=str(PROJECT_ROOT / "mlruns"))
    Path(PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / "sequence_rul_leaderboard.json").write_text(
        json.dumps(board, indent=2))
