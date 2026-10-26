"""Module 2 - real vs decoy filtering.

Reads Module 1 detections, merges duplicate boxes on one object, measures
depth geometry inside every box and decides real vs printed/displayed decoy.
A print is flat (one plane); a real extinguisher is a cylinder (~15 cm across)
with a hose and handle, so depth varies across its box.

Usage:
    python src/decoy_filter.py --split test
    python src/decoy_filter.py --split eval
Writes outputs/filtered_<split>.json: {"meta": {...}, "images": [Module 1 entries
with per detection is_real, decoy_score, features, depth_ok, depth_gap_ms, merged_from]}.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from depth_utils import DepthIndex, backproject, load_depth_m, split_intrinsics, timestamp

# ---- settings ----
MERGE_IOU = 0.5           # merge two boxes in one frame when IoU above this ...
MERGE_CONTAIN = 0.8       # ... or intersection / smaller-box area above this
MAX_GAP_MS = 100.0        # reject depth frames further than this from the colour frame
INNER_X, INNER_Y = 0.15, 0.10   # shrink box by this fraction per side (drop mixed edge pixels)
Z_MIN, Z_MAX = 0.2, 8.0   # valid depth range, metres
MIN_VALID_FRAC = 0.3      # below this the box depth is not trusted
RING_PX = 8               # width of the background ring around the box


# ---------------- duplicate merge ----------------
def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def merge_duplicates(dets):
    """Greedy, highest confidence first: absorb boxes that overlap or are contained."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        for k in kept:
            i = _inter(d["bbox_xyxy"], k["bbox_xyxy"])
            union = _area(d["bbox_xyxy"]) + _area(k["bbox_xyxy"]) - i
            small = min(_area(d["bbox_xyxy"]), _area(k["bbox_xyxy"]))
            if (union > 0 and i / union > MERGE_IOU) or (small > 0 and i / small > MERGE_CONTAIN):
                k["merged_from"].append(d["det_id"])
                break
        else:
            kept.append({**d, "merged_from": []})
    return kept


# ---------------- depth features ----------------
def _inner(box, W, H):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * INNER_X, (y2 - y1) * INNER_Y
    return (int(max(0, x1 + dx)), int(max(0, y1 + dy)),
            int(min(W, np.ceil(x2 - dx))), int(min(H, np.ceil(y2 - dy))))


def _plane_rms(P):
    """RMS distance (m) of 3D points to their least-squares plane, after one 2.5-sigma trim."""
    for _ in range(2):
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        r = (P - c) @ vt[2]
        rms = np.sqrt(np.mean(r ** 2))
        keep = np.abs(r) < 2.5 * rms + 1e-6
        if keep.sum() < 20 or keep.all():
            break
        P = P[keep]
    return float(rms)


def box_features(box, z, K):
    """Depth-geometry features for one box. Returns dict (NaN where not computable)."""
    H, W = z.shape
    x1, y1, x2, y2 = _inner(box, W, H)
    nan = float("nan")
    f = {"valid_frac": 0.0, "z_med": nan, "plane_rms_mm": nan, "plane_rms_norm": nan,
         "bulge_mm": nan, "z_iqr_mm": nan, "fg_step_mm": nan, "height_m": nan, "width_m": nan}
    if x2 - x1 < 3 or y2 - y1 < 3:
        return f
    patch = z[y1:y2, x1:x2]
    ok = np.isfinite(patch) & (patch > Z_MIN) & (patch < Z_MAX)
    f["valid_frac"] = float(ok.mean())
    if ok.sum() < 30:
        return f
    zm = float(np.median(patch[ok]))
    f["z_med"] = zm
    vs, us = np.nonzero(ok)
    P = backproject(us + x1, vs + y1, patch[ok], K)
    rms = _plane_rms(P)
    f["plane_rms_mm"] = rms * 1000
    f["plane_rms_norm"] = rms * 1000 / zm ** 2       # depth noise grows ~ z^2
    q75, q25 = np.percentile(patch[ok], [75, 25])
    f["z_iqr_mm"] = float(q75 - q25) * 1000
    # cylinder bulge: centre third of columns closer than the outer thirds
    cols = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan
                     for c in np.where(ok, patch, np.nan).T])
    n = len(cols)
    if n >= 6:
        mid, side = cols[n // 3: 2 * n // 3], np.r_[cols[: n // 3], cols[2 * n // 3:]]
        if np.isfinite(mid).any() and np.isfinite(side).any():
            f["bulge_mm"] = float(np.nanmedian(side) - np.nanmedian(mid)) * 1000
    # background step: ring just outside the full box minus box interior
    bx1, by1, bx2, by2 = [int(round(v)) for v in box]
    ox1, oy1, ox2, oy2 = max(0, bx1 - RING_PX), max(0, by1 - RING_PX), min(W, bx2 + RING_PX), min(H, by2 + RING_PX)
    ring = z[oy1:oy2, ox1:ox2].copy()
    ring[max(0, by1) - oy1: by2 - oy1, max(0, bx1) - ox1: bx2 - ox1] = np.nan
    ring = ring[np.isfinite(ring) & (ring > Z_MIN) & (ring < Z_MAX)]
    if ring.size > 20:
        f["fg_step_mm"] = float(np.median(ring) - zm) * 1000
    fx, fy = K[0, 0], K[1, 1]
    f["height_m"] = (box[3] - box[1]) * zm / fy
    f["width_m"] = (box[2] - box[0]) * zm / fx
    return f


def extract(images, raw_dir):
    """Merge duplicates and attach depth features to every detection (no decision yet)."""
    K, W, H = split_intrinsics(raw_dir)
    idx = DepthIndex(raw_dir)
    out, n_merged, t_feat = [], 0, 0.0
    for im in images:
        dets = merge_duplicates(im["detections"])
        n_merged += len(im["detections"]) - len(dets)
        dpath, gap = idx.match(timestamp(im["image_id"]), MAX_GAP_MS)
        t0 = time.perf_counter()
        z = load_depth_m(dpath) if dpath is not None else None
        for d in dets:
            d["depth_frame"] = dpath.name if dpath is not None else None
            d["depth_gap_ms"] = round(gap, 1)
            d["features"] = box_features(d["bbox_xyxy"], z, K) if z is not None else {"valid_frac": 0.0}
            d["depth_ok"] = bool(z is not None and d["features"]["valid_frac"] >= MIN_VALID_FRAC)
        t_feat += time.perf_counter() - t0
        out.append({**im, "detections": dets})
    stats = {"n_merged": n_merged, "feature_ms_per_frame": 1000 * t_feat / max(1, len(images))}
    return out, stats


# ---------------- decision ----------------
# Logistic regression on two depth features, fitted on the dev (test) recording:
#   log(plane_rms_norm)  - flat print -> small, cylinder + hose -> large
#   height_m             - physical box height; prints are ~0.4 m, real objects ~1.2 m (with hose)
FEATURES = ["log_plane_rms_norm", "height_m"]
P_REAL_MIN = 0.5          # accept as real when P(real) >= this, i.e. decoy_score < 0.5
FALLBACK_CONF = 0.85      # no usable depth: accept only confident detections (dev: TPR 0.81, FPR 0.05)
MODEL_PATH = "models/decoy_filter.json"


def feature_vector(d):
    f = d["features"]
    return [np.log(max(f["plane_rms_norm"], 0.1)), f["height_m"]]


def usable(d):
    return d["depth_ok"] and all(np.isfinite(feature_vector(d)))


def fit(X, y):
    """Standardise + logistic regression; returns plain-number params (no pickle)."""
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0)
    lr = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
    return {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def p_real(model, x):
    z = (np.asarray(x) - model["mean"]) / np.asarray(model["std"])
    return float(1 / (1 + np.exp(-(z @ np.asarray(model["coef"]) + model["intercept"]))))


def decide(images, model):
    """Fill is_real and decoy_score (= 1 - P(real)) for every detection."""
    for im in images:
        for d in im["detections"]:
            if usable(d):
                d["decoy_score"] = round(1 - p_real(model, feature_vector(d)), 4)
                d["is_real"] = d["decoy_score"] <= 1 - P_REAL_MIN
                d["decision"] = "depth"
            else:
                d["decoy_score"] = round(1 - d["confidence"], 4)
                d["is_real"] = d["confidence"] >= FALLBACK_CONF
                d["decision"] = "confidence_fallback"
    return images


def training_rows(images, labels):
    """(image_id, detection, 'real'|'decoy') for detections whose centre lies in exactly one class."""
    from labels import centre_in
    rows = []
    for im in images:
        for d in im["detections"]:
            cls = {o["cls"] for o in labels.get(im["image_id"], []) if centre_in(d["bbox_xyxy"], o["box"])}
            if len(cls) == 1 and usable(d):
                rows.append((im["image_id"], d, cls.pop()))
    return rows


def block_cv(images, labels, n_blocks=5):
    """Out-of-fold metric over contiguous frame blocks (neighbouring frames are near-duplicates)."""
    from labels import evaluate
    ids = sorted(im["image_id"] for im in images)
    block = {k: i * n_blocks // len(ids) for i, k in enumerate(ids)}
    rows = training_rows(images, labels)
    for b in range(n_blocks):
        m = fit([feature_vector(d) for i, d, c in rows if block[i] != b],
                [c == "real" for i, d, c in rows if block[i] != b])
        decide([im for im in images if block[im["image_id"]] == b], m)
    r = evaluate(images, labels)
    r.pop("objects")
    return r


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from labels import evaluate, load_labels

    p = argparse.ArgumentParser(description="Module 2: real vs decoy filtering")
    p.add_argument("--split", required=True, choices=["eval", "test"])
    p.add_argument("--dets", help="default outputs/detections_<split>.json")
    p.add_argument("--raw", help="default data/raw/<split>")
    p.add_argument("--out", help="default outputs/filtered_<split>.json")
    p.add_argument("--fit", nargs="+", metavar="LABEL_DIR",
                   help="fit the decision model on these labels (dev only) and save it to --model")
    p.add_argument("--labels", nargs="+", metavar="LABEL_DIR", help="report TPR/FPR against these labels")
    p.add_argument("--model", default=MODEL_PATH)
    a = p.parse_args()
    dets = a.dets or f"outputs/detections_{a.split}.json"
    raw = a.raw or f"data/raw/{a.split}"
    out = a.out or f"outputs/filtered_{a.split}.json"

    images = json.load(open(dets))
    images, stats = extract(images, raw)
    print(f"[decoy_filter] {a.split}: {len(images)} frames, merged {stats['n_merged']} duplicate boxes, "
          f"features {stats['feature_ms_per_frame']:.1f} ms/frame")

    if a.fit:
        labels = load_labels(a.fit)
        cv = block_cv(images, labels)
        print(f"[decoy_filter] 5-block CV on {a.split}: TPR {cv['TPR']:.3f} ({cv['real_accepted']}/{cv['n_real']}), "
              f"FPR {cv['FPR']:.3f} ({cv['decoy_accepted']}/{cv['n_decoy']})")
        rows = training_rows(images, labels)
        model = fit([feature_vector(d) for _, d, _ in rows], [c == "real" for _, _, c in rows])
        model.update({"fitted_on": a.fit, "n_train": len(rows),
                      "n_real": sum(c == "real" for _, _, c in rows), "cv": cv})
        Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        json.dump(model, open(a.model, "w"), indent=1)
        print(f"[decoy_filter] model saved to {a.model}")
    model = json.load(open(a.model))

    t0 = time.perf_counter()
    decide(images, model)
    decide_ms = 1000 * (time.perf_counter() - t0) / max(1, len(images))
    n = [d for im in images for d in im["detections"]]
    print(f"[decoy_filter] {sum(d['is_real'] for d in n)}/{len(n)} detections accepted as real, "
          f"{sum(d['decision'] == 'confidence_fallback' for d in n)} used the confidence fallback")

    meta = {"module": 2, "split": a.split, "detections": dets, "model": a.model,
            "decision": f"is_real = P(real) >= {P_REAL_MIN} (decoy_score <= {1 - P_REAL_MIN}); "
                        f"without usable depth: confidence >= {FALLBACK_CONF}",
            "p_real_min": P_REAL_MIN, "fallback_conf": FALLBACK_CONF,
            "merge": {"iou": MERGE_IOU, "contain": MERGE_CONTAIN, "n_merged": stats["n_merged"]},
            "depth": {"max_gap_ms": MAX_GAP_MS, "inner_shrink_xy": [INNER_X, INNER_Y],
                      "valid_range_m": [Z_MIN, Z_MAX], "min_valid_frac": MIN_VALID_FRAC},
            "timing_ms_per_frame": {"features": round(stats["feature_ms_per_frame"], 2),
                                    "decision": round(decide_ms, 3)}}
    if a.labels:
        r = evaluate(images, load_labels(a.labels))
        r.pop("objects")
        meta["metrics"] = r
        print(f"[decoy_filter] {a.split}: TPR {r['TPR']:.3f} ({r['real_accepted']}/{r['n_real']}), "
              f"FPR {r['FPR']:.3f} ({r['decoy_accepted']}/{r['n_decoy']}) on {r['frames']} labelled frames")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "images": images}, open(out, "w"), indent=1)
    print(f"[decoy_filter] wrote {out}")


if __name__ == "__main__":
    main()
