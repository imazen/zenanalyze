#!/usr/bin/env python3
"""MLP classifier picker over v05c — closer to deployable than the
HistGradientBoostingClassifier prototype.

Same teacher signal as picker_v06_proto.py (safety-constrained label per
(image, distance)) but trained as a small PyTorch MLP that can be
serialized into ZNPR-compatible format. Input is (features +
log(distance)); output is a 5-class softmax over (effort1, effort3,
effort5, effort7, effort9).

Bypasses train_hybrid.py's regress-bytes-then-argmin chain which the
safety mask doesn't play well with.
"""
from __future__ import annotations
import csv, json, sys, math, random
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

FEATURES_TSV = Path(
    "/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv"
)
SWEEP_TSV = Path("/home/lilith/sweep-data/zenjxl_v05c.tsv")
DEFAULT_EFFORT = 7
EFFORTS = [1, 3, 5, 7, 9]
EFFORT_TO_IDX = {e: i for i, e in enumerate(EFFORTS)}

ZENSIM_TOL = 0.05
SPEED_TOL = 1.05
BYTES_GAIN = 0.99


def load_features():
    feats = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        keep = [c for c in rdr.fieldnames if c.startswith("feat_")]
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in keep]
                feats[r["image_path"]] = vec
            except Exception:
                continue
    return feats, keep


def load_sweep():
    rows = []
    with open(SWEEP_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                if not r["encoded_bytes"] or not r["score_zensim"]:
                    continue
                k = json.loads(r["knob_tuple_json"])
                if k.get("noise") is True:
                    continue
                rows.append(
                    (
                        r["image_path"].rsplit("/", 1)[-1],
                        round(float(k["distance"]), 4),
                        int(k["effort"]),
                        int(r["encoded_bytes"]),
                        float(r["encode_ms"]),
                        float(r["score_zensim"]),
                    )
                )
            except Exception:
                continue
    return rows


def build_dataset(feats, sweep_rows):
    by_id = defaultdict(dict)
    for img, d, e, b, ms, z in sweep_rows:
        by_id[(img, d)][e] = (b, ms, z)

    X, y, raw = [], [], []
    for (img, dist), cells in by_id.items():
        if img not in feats or DEFAULT_EFFORT not in cells:
            continue
        b_def, ms_def, z_def = cells[DEFAULT_EFFORT]
        best_label = EFFORT_TO_IDX[DEFAULT_EFFORT]
        best_bytes = b_def
        for e in EFFORTS:
            if e == DEFAULT_EFFORT or e not in cells:
                continue
            b, ms, z = cells[e]
            if (
                b < best_bytes * BYTES_GAIN
                and z >= z_def - ZENSIM_TOL
                and ms <= ms_def * SPEED_TOL
            ):
                if b < best_bytes:
                    best_bytes = b
                    best_label = EFFORT_TO_IDX[e]
        x = feats[img] + [math.log(max(dist, 0.01))]
        X.append(x)
        y.append(best_label)
        raw.append((img, dist, cells))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), raw


def main():
    feats, feat_cols = load_features()
    print(f"[features] {len(feats)} images, {len(feat_cols)} feats", file=sys.stderr)

    sweep_rows = load_sweep()
    print(f"[sweep] {len(sweep_rows)} cells", file=sys.stderr)

    X, y, raw = build_dataset(feats, sweep_rows)
    print(f"[teacher] {len(X)} cells, label dist: {Counter(EFFORTS[lab] for lab in y)}", file=sys.stderr)

    # Normalize features
    mean_x = X.mean(axis=0)
    std_x = X.std(axis=0) + 1e-9
    X_n = (X - mean_x) / std_x

    # Image-level split
    rng = random.Random(7)
    all_imgs = sorted({img for img, _, _ in raw})
    rng.shuffle(all_imgs)
    n_holdout = int(len(all_imgs) * 0.2)
    holdout = set(all_imgs[:n_holdout])
    tr_idx = np.array([i for i, (img, _, _) in enumerate(raw) if img not in holdout])
    va_idx = np.array([i for i, (img, _, _) in enumerate(raw) if img in holdout])

    print(f"[split] train={len(tr_idx)} val={len(va_idx)}", file=sys.stderr)

    # Tiny PyTorch MLP classifier
    import torch
    import torch.nn as nn
    torch.manual_seed(7)

    n_in = X.shape[1]
    n_classes = len(EFFORTS)

    net = nn.Sequential(
        nn.Linear(n_in, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )

    # Class-balanced loss
    label_counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
    class_weights = (label_counts.sum() / (label_counts + 1.0)).clip(0.5, 5.0)
    print(f"[loss] class_weights: {dict(zip(EFFORTS, class_weights.round(2).tolist()))}", file=sys.stderr)

    cw = torch.tensor(class_weights)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    X_tr_t = torch.from_numpy(X_n[tr_idx])
    y_tr_t = torch.from_numpy(y[tr_idx])
    X_va_t = torch.from_numpy(X_n[va_idx])
    y_va_t = torch.from_numpy(y[va_idx])

    BATCH = 512
    EPOCHS = 60
    n_train = X_tr_t.shape[0]

    best_va_acc = 0.0
    best_state = None
    bad_epochs = 0
    for epoch in range(EPOCHS):
        perm = torch.randperm(n_train)
        net.train()
        total_loss = 0.0
        for s in range(0, n_train, BATCH):
            ix = perm[s:s + BATCH]
            opt.zero_grad()
            logits = net(X_tr_t[ix])
            loss = loss_fn(logits, y_tr_t[ix])
            loss.backward()
            opt.step()
            total_loss += float(loss) * ix.shape[0]
        total_loss /= n_train

        net.eval()
        with torch.no_grad():
            tr_pred = net(X_tr_t).argmax(dim=1)
            va_pred = net(X_va_t).argmax(dim=1)
            tr_acc = float((tr_pred == y_tr_t).float().mean())
            va_acc = float((va_pred == y_va_t).float().mean())
        if va_acc > best_va_acc + 1e-4:
            best_va_acc = va_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:3d}  loss {total_loss:.4f}  tr_acc {tr_acc:.3f}  va_acc {va_acc:.3f}", file=sys.stderr)
        if bad_epochs >= 15:
            print(f"[early-stop] no improvement for 15 epochs", file=sys.stderr)
            break

    net.load_state_dict(best_state)
    net.eval()

    # Final predictions on val
    with torch.no_grad():
        va_pred = net(X_va_t).argmax(dim=1).numpy()
    print(f"\n[predict] val pred dist: {Counter(EFFORTS[p] for p in va_pred)}", file=sys.stderr)
    print(f"[predict] val truth dist: {Counter(EFFORTS[t] for t in y_va_t.numpy())}", file=sys.stderr)
    print(f"[final] val acc: {best_va_acc:.3f}", file=sys.stderr)

    # A/B against effort=7 default
    dbytes, dzensim, dms = [], [], []
    n_default = 0
    n_alt = 0
    safe_dbytes_only = []
    safe_dzensim_only = []
    safe_dms_only = []
    band_data = defaultdict(lambda: {"db": [], "dz": [], "dm": []})
    for k, i in enumerate(va_idx):
        img, dist, cells = raw[i]
        picked_e = EFFORTS[va_pred[k]]
        if picked_e not in cells or DEFAULT_EFFORT not in cells:
            continue
        b_def, ms_def, z_def = cells[DEFAULT_EFFORT]
        b, ms, z = cells[picked_e]
        d_pct = 100 * (b - b_def) / b_def
        dz_pp = z - z_def
        dm_pct = 100 * (ms - ms_def) / ms_def
        dbytes.append(d_pct); dzensim.append(dz_pp); dms.append(dm_pct)
        if picked_e == DEFAULT_EFFORT:
            n_default += 1
        else:
            n_alt += 1
            safe_dbytes_only.append(d_pct)
            safe_dzensim_only.append(dz_pp)
            safe_dms_only.append(dm_pct)
        band = "tight" if dist <= 1.0 else ("mid" if dist <= 3.0 else "loose")
        band_data[band]["db"].append(d_pct)
        band_data[band]["dz"].append(dz_pp)
        band_data[band]["dm"].append(dm_pct)

    print(f"\n## A/B: MLP classifier picker on v05c held-out {len(holdout)} images, {len(dbytes)} cells")
    print(f"   Default: effort=7, noise=False")
    print(f"   Picker decisions: {n_default} default ({100*n_default/len(dbytes):.1f}%), "
          f"{n_alt} safe-alt ({100*n_alt/len(dbytes):.1f}%)")
    print(f"\n   Bytes vs default:  mean {mean(dbytes):+.3f}%  median {median(dbytes):+.3f}%")
    print(f"   Zensim vs default: mean {mean(dzensim):+.4f}pp")
    print(f"   Encode ms vs def:  mean {mean(dms):+.2f}%")
    if safe_dbytes_only:
        print(f"\n   ON CELLS WHERE PICKER DEPARTS FROM DEFAULT ({len(safe_dbytes_only)}):")
        print(f"     mean Δbytes:  {mean(safe_dbytes_only):+.3f}%")
        print(f"     mean Δzensim: {mean(safe_dzensim_only):+.4f}pp")
        print(f"     mean Δms:     {mean(safe_dms_only):+.2f}%")
    print(f"\n## Per-distance band")
    print(f"{'band':10s} {'n':>5s} {'mean Δbytes':>14s} {'mean Δzensim':>14s} {'mean Δms':>10s}")
    for b in ["tight", "mid", "loose"]:
        d = band_data[b]
        if d["db"]:
            print(f"{b:10s} {len(d['db']):>5d} {mean(d['db']):>+14.3f}% {mean(d['dz']):>+14.4f}pp {mean(d['dm']):>+10.2f}%")

    # Verdict
    if mean(dbytes) < -0.5 and mean(dzensim) > -0.05:
        print("\n[verdict] **SHIP** — meaningful bytes savings without quality regression")
    elif mean(dbytes) < 0.0 and mean(dzensim) > 0.0:
        print("\n[verdict] **MARGINAL SHIP** — small bytes savings + quality bonus")
    else:
        print("\n[verdict] HOLD")


if __name__ == "__main__":
    main()
