"""Module 2 - real vs decoy filtering.

A printed decoy is flat and small; a real extinguisher (cylinder + hose) is not
flat and about 1 m tall in its box. Depth is registered to the colour camera
(same K, uint16 mm), so colour boxes index the depth image directly.

Usage:
    python src/decoy_filter.py --split test --labels labels/chris_test
    python src/decoy_filter.py --split eval
Writes outputs/filtered_<split>.json: Module 1 entries, each detection with
is_real, decoy_score (0 real, 1 decoy), flatness, height_m, z_med, depth_gap_ms.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

FLAT_MIN = 5.0         # plane-fit RMS (mm) / z^2 (m^2) above this = not flat
HEIGHT_MIN = 0.7       # physical box height (m) above this = extinguisher size
FALLBACK_CONF = 0.85   # no usable depth: accept only confident detections
MAX_GAP_MS = 100       # max time between colour frame and matched depth frame


def ts(name):
    return int(re.search(r"(\d{19})", name).group(1))


def merge_duplicates(dets):
    """Drop a box whose area is mostly (>80 %) inside a more confident box."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        a = d["bbox_xyxy"]
        area = (a[2] - a[0]) * (a[3] - a[1])
        if not any(max(0, min(a[2], k["bbox_xyxy"][2]) - max(a[0], k["bbox_xyxy"][0])) *
                   max(0, min(a[3], k["bbox_xyxy"][3]) - max(a[1], k["bbox_xyxy"][1])) > 0.8 * area
                   for k in kept):
            kept.append(d)
    return kept


def features(box, z, K):
    """flatness, height_m, z_med from the inner box (15 % / 10 % edges trimmed)."""
    x1, y1, x2, y2 = box
    dx, dy = 0.15 * (x2 - x1), 0.10 * (y2 - y1)
    x1, y1, x2, y2 = int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy)
    patch = z[y1:y2, x1:x2]
    v, u = np.nonzero((patch > 0.2) & (patch < 8))
    if len(v) < 30 or len(v) < 0.3 * patch.size:
        return None
    d = patch[v, u]
    P = np.stack([(u + x1 - K[0, 2]) * d / K[0, 0], (v + y1 - K[1, 2]) * d / K[1, 1], d], 1)
    P -= P.mean(0)
    rms = np.linalg.svd(P, full_matrices=False)[1][2] / np.sqrt(len(P))   # distance to best plane
    zm = float(np.median(d))
    return {"flatness": round(float(rms) * 1000 / zm ** 2, 2),
            "height_m": round((box[3] - box[1]) * zm / K[1, 1], 3), "z_med": round(zm, 3)}


def run(split):
    raw = Path("data/raw", split)
    info = next((raw / "camera_color_camera_info").glob("*.txt")).read_text()
    K = np.array([float(v) for v in re.search(r"K: \[(.*?)\]", info).group(1).split(",")]).reshape(3, 3)
    depth = sorted((raw / "camera_depth_image_raw").glob("*.png"), key=lambda p: ts(p.name))
    images = json.load(open(f"outputs/detections_{split}.json"))
    for im in images:
        dpath = min(depth, key=lambda p: abs(ts(p.name) - ts(im["image_id"])))
        gap = abs(ts(dpath.name) - ts(im["image_id"])) / 1e6
        z = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED) / 1000.0 if gap <= MAX_GAP_MS else None
        im["detections"] = merge_duplicates(im["detections"])
        for d in im["detections"]:
            f = features(d["bbox_xyxy"], z, K) if z is not None else None
            if f:
                d.update(f)
                d["is_real"] = bool(f["flatness"] > FLAT_MIN and f["height_m"] > HEIGHT_MIN)
            else:
                d["is_real"] = d["confidence"] >= FALLBACK_CONF
            d["decoy_score"] = 0.0 if d["is_real"] else 1.0
            d["depth_gap_ms"] = round(gap, 1)
    return images


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["eval", "test"])
    p.add_argument("--labels", nargs="+", help="label folders: print TPR/FPR")
    a = p.parse_args()
    images = run(a.split)
    json.dump(images, open(f"outputs/filtered_{a.split}.json", "w"))
    n = [d for im in images for d in im["detections"]]
    print(f"{a.split}: {sum(d['is_real'] for d in n)}/{len(n)} detections accepted as real")
    if a.labels:
        from labels import evaluate, load_labels
        r = evaluate(images, load_labels(a.labels))
        print(f"TPR {r['TPR']:.3f} ({r['real_accepted']}/{r['n_real']}), "
              f"FPR {r['FPR']:.3f} ({r['decoy_accepted']}/{r['n_decoy']})")
