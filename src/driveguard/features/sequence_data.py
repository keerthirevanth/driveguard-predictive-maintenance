"""Milestone 4 (part 2) - build raw SMART sequence windows for deep RUL models.

Unlike the tabular feature sets (which summarise history into rolling stats), sequence
models want the raw trajectory. For sampled (drive, day) anchor points we extract the last
L days of the big5 SMART values -> a tensor of shape [N, L, F], with survival targets
(event, duration). Shorter histories are left-zero-padded.

To stay tractable we keep all failing drives (dense anchors near end-of-life) plus a sample
of censored drives (a few anchors each), then cap the total anchors per split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from driveguard.features.build_features import SMART_BIG5, _QUARTER_RANGE


def build_sequences(interim_glob: str, summary_path: str, quarters: list[str],
                    L: int = 30, cap_event_drives: int = 5_000,
                    cap_censored_drives: int = 8_000,
                    cap_anchors: int = 120_000, event_anchors_per_drive: int = 30,
                    censored_anchors_per_drive: int = 4, seed: int = 42):
    lo = min(_QUARTER_RANGE[q][0] for q in quarters)
    hi = max(_QUARTER_RANGE[q][1] for q in quarters)
    lo_d = np.datetime64(lo)
    hi_d = np.datetime64(hi)
    rng = np.random.default_rng(seed)

    summ = pl.read_parquet(summary_path).select(
        "serial_number", "first_date", "last_date", "event")
    active = summ.filter(
        (pl.col("first_date") <= pl.lit(hi).str.to_date())
        & (pl.col("last_date") >= pl.lit(lo).str.to_date()))
    ev = active.filter(pl.col("event") == 1)
    cen = active.filter(pl.col("event") == 0)
    if ev.height > cap_event_drives:
        ev = ev.sample(n=cap_event_drives, seed=seed)
    if cen.height > cap_censored_drives:
        cen = cen.sample(n=cap_censored_drives, seed=seed)
    keep = pl.concat([ev, cen])
    print(f"  [seq] drives: {ev.height} event + {cen.height} censored", flush=True)
    keep_set = set(keep["serial_number"].to_list())
    ends = dict(zip(keep["serial_number"], keep["last_date"]))
    evmap = dict(zip(keep["serial_number"], keep["event"]))

    lookback = (pl.lit(lo).str.to_date() - pl.duration(days=L))
    df = (
        pl.scan_parquet(interim_glob)
        .select(["date", "serial_number", *SMART_BIG5])
        .filter(pl.col("serial_number").is_in(list(keep_set)))
        .filter(pl.col("date").is_between(lookback, pl.lit(hi).str.to_date()))
        .with_columns([pl.col(c).cast(pl.Float32) for c in SMART_BIG5])
        .sort("serial_number", "date")
        .with_columns([
            pl.col(c).forward_fill().over("serial_number").fill_null(0).alias(c)
            for c in SMART_BIG5
        ])
        .collect(engine="streaming")
    )
    print(f"  [seq] collected {df.height} drive-days; building windows...", flush=True)

    X, dur, event = [], [], []
    F = len(SMART_BIG5)
    for serial, g in df.group_by("serial_number", maintain_order=True):
        serial = serial[0] if isinstance(serial, tuple) else serial
        dates = g["date"].to_numpy()
        feats = g.select(SMART_BIG5).to_numpy().astype(np.float32)  # [days, F]
        last = np.datetime64(ends[serial])
        ev_flag = int(evmap[serial])
        # anchor positions = days inside the split window
        anchors = np.where((dates >= lo_d) & (dates <= hi_d))[0]
        if len(anchors) == 0:
            continue
        if ev_flag == 1:
            anchors = anchors[-event_anchors_per_drive:]
        elif len(anchors) > censored_anchors_per_drive:
            anchors = rng.choice(anchors, censored_anchors_per_drive, replace=False)
        for i in anchors:
            w = feats[max(0, i - L + 1): i + 1]
            if len(w) < L:  # left-pad short histories with zeros
                w = np.vstack([np.zeros((L - len(w), F), np.float32), w])
            X.append(w)
            rul = (last - dates[i]) / np.timedelta64(1, "D")
            dur.append(max(float(rul), 1.0))
            event.append(ev_flag)

    X = np.asarray(X, dtype=np.float32)
    dur = np.asarray(dur, dtype=np.float32)
    event = np.asarray(event, dtype=bool)
    # global cap with a balanced event/censored mix (both are needed: events carry the
    # RUL signal, censored teach the "survives at least this long" constraint + C-index)
    ev_idx, ce_idx = np.where(event)[0], np.where(~event)[0]
    n_ce_keep = min(len(ce_idx), cap_anchors // 2)
    n_ev_keep = min(len(ev_idx), cap_anchors - n_ce_keep)
    n_ce_keep = min(len(ce_idx), cap_anchors - n_ev_keep)  # backfill if few events
    if len(ev_idx) > n_ev_keep:
        ev_idx = rng.choice(ev_idx, n_ev_keep, replace=False)
    if len(ce_idx) > n_ce_keep:
        ce_idx = rng.choice(ce_idx, n_ce_keep, replace=False)
    sel = np.concatenate([ev_idx, ce_idx])
    rng.shuffle(sel)
    return X[sel], event[sel], dur[sel]


def build_and_save(cfg: dict, project_root: Path, L: int = 30) -> dict:
    interim = project_root / cfg["data"]["interim_dir"]
    processed = project_root / cfg["data"]["processed_dir"]
    out = processed / "sequences"
    out.mkdir(parents=True, exist_ok=True)
    glob = str(interim / "*.parquet")
    summ = str(processed / "drive_summary.parquet")
    info = {}
    for split, quarters in [("train", cfg["split"]["train_quarters"]),
                            ("val", cfg["split"]["val_quarters"]),
                            ("test", cfg["split"]["test_quarters"])]:
        X, e, d = build_sequences(glob, summ, quarters, L=L)
        np.savez_compressed(out / f"{split}.npz", X=X, event=e, dur=d)
        info[split] = {"n": int(len(X)), "events": int(e.sum()), "shape": list(X.shape)}
    info["out_dir"] = str(out)
    return info


if __name__ == "__main__":
    import json

    from driveguard.config import PROJECT_ROOT, load_config

    print(json.dumps(build_and_save(load_config(), PROJECT_ROOT), indent=2))
