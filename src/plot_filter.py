"""Module 2 - figures for the slides (dev = test recording).

Usage:
    python src/plot_filter.py --filtered outputs/filtered_test.json --labels labels/chris_test
Writes outputs/figures/: feature_scatter.png, threshold_sweep.png, frame_*.png
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decoy_filter import P_REAL_MIN, feature_vector, usable
from depth_utils import load_depth_m
from labels import centre_in, evaluate, load_labels

REAL, DECOY = "#2a78d6", "#eb6834"      # categorical slots 1-2 (blue = real, orange = decoy)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def matched_rows(images, labels):
    rows = []
    for im in images:
        for d in im["detections"]:
            cls = {o["cls"] for o in labels.get(im["image_id"], []) if centre_in(d["bbox_xyxy"], o["box"])}
            if len(cls) == 1 and usable(d):
                rows.append((feature_vector(d), cls.pop()))
    return rows


def feature_scatter(images, labels, model, out):
    rows = matched_rows(images, labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(color=GRID, lw=0.8)
    for cls, col in [("decoy", DECOY), ("real", REAL)]:
        X = np.array([x for x, c in rows if c == cls])
        ax.scatter(X[:, 0], X[:, 1], s=28, color=col, edgecolor="white", lw=1,
                   label=f"{cls} ({len(X)} detections)", zorder=3)
    # decision boundary P(real) = P_REAL_MIN in the original feature units
    mu, sd, w, b = map(np.asarray, (model["mean"], model["std"], model["coef"], model["intercept"]))
    logit = np.log(P_REAL_MIN / (1 - P_REAL_MIN))
    xs = np.linspace(*ax.get_xlim(), 50)
    ys = mu[1] + sd[1] * (logit - b - w[0] * (xs - mu[0]) / sd[0]) / w[1]
    ax.plot(xs, ys, color=INK, lw=1.5, ls="--", label=f"decision boundary, P(real) = {P_REAL_MIN}")
    ax.set_ylim(0, max(1.8, ax.get_ylim()[1]))
    ax.set_xlabel("log flatness: plane-fit RMS / z²  (mm / m²)")
    ax.set_ylabel("physical box height (m)")
    ax.set_title("Dev set: decoys are flat and small", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "feature_scatter.png")
    plt.close(fig)


def threshold_sweep(images, labels, out):
    ts = np.linspace(0.02, 0.98, 49)
    tpr, fpr = [], []
    for t in ts:
        for im in images:
            for d in im["detections"]:
                d["_keep"] = d["is_real"]
                if d["decision"] == "depth":
                    d["is_real"] = (1 - d["decoy_score"]) >= t
        r = evaluate(images, labels)
        tpr.append(r["TPR"]); fpr.append(r["FPR"])
        for im in images:
            for d in im["detections"]:
                d["is_real"] = d.pop("_keep")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.grid(color=GRID, lw=0.8)
    ax.plot(ts, tpr, color=REAL, lw=2, label="real-object TPR")
    ax.plot(ts, fpr, color=DECOY, lw=2, label="decoy FPR")
    ax.axhline(0.75, color=REAL, lw=1, ls=":")
    ax.axhline(0.15, color=DECOY, lw=1, ls=":")
    ax.text(0.99, 0.76, "TPR target 0.75", ha="right", va="bottom", color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.text(0.99, 0.16, "FPR target 0.15", ha="right", va="bottom", color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvline(P_REAL_MIN, color=INK, lw=1, ls="--")
    ax.text(P_REAL_MIN + 0.01, 0.45, f"chosen: P(real) ≥ {P_REAL_MIN}", color=INK, fontsize=8)
    ax.set_xlabel("acceptance threshold on P(real)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Dev set (in-sample): TPR and FPR vs threshold", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png")
    plt.close(fig)


def annotate(im, raw_dir, labels, out_path, title):
    img = cv2.imread(im["image_path"]) if Path(im["image_path"]).exists() else \
        cv2.imread(str(Path(raw_dir, "camera_color_image_raw", im["image_id"] + ".png")))
    for o in labels.get(im["image_id"], []):           # reference boxes, thin grey
        x1, y1, x2, y2 = map(int, o["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
    depth_name = next((d["depth_frame"] for d in im["detections"] if d.get("depth_frame")), None)
    if depth_name:
        z = load_depth_m(Path(raw_dir, "camera_depth_image_raw", depth_name))
        zc = cv2.applyColorMap((np.clip(np.nan_to_num(z, nan=0) / 6, 0, 1) * 255).astype(np.uint8),
                               cv2.COLORMAP_VIRIDIS)
        zc[~np.isfinite(z)] = 0
    else:
        zc = np.zeros_like(img)
    for d in im["detections"]:
        x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
        col = (214, 120, 42) if d["is_real"] else (52, 104, 235)   # BGR of REAL / DECOY
        tag = f"{'REAL' if d['is_real'] else 'DECOY'} {1 - d['decoy_score']:.2f}"
        for canvas in (img, zc):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
            cv2.putText(canvas, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
            cv2.putText(canvas, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    both = np.hstack([img, zc])
    cv2.putText(both, title, (8, both.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), both)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filtered", default="outputs/filtered_test.json")
    p.add_argument("--labels", nargs="+", default=["labels/chris_test"])
    p.add_argument("--raw", default="data/raw/test")
    p.add_argument("--out", default="outputs/figures")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(a.filtered))
    images, model = data["images"], json.load(open(data["meta"]["model"]))
    labels = load_labels(a.labels)
    feature_scatter(images, labels, model, out)
    threshold_sweep(images, labels, out)

    r = evaluate(images, labels)
    by_id = {im["image_id"]: im for im in images}
    # a frame where every real object is accepted and every detected decoy rejected, with most decoys
    def clean(k):
        objs = [o for o in r["objects"] if o["image_id"] == k]
        return all(o["accepted"] == (o["cls"] == "real") for o in objs if o["detected"])
    good = sorted((k for k in labels if clean(k)),
                  key=lambda k: -sum(o["detected"] and o["cls"] == "decoy" for o in r["objects"] if o["image_id"] == k))
    if good:
        annotate(by_id[good[0]], a.raw, labels, out / "frame_correct.png", "real accepted, decoys rejected")
    for kind, sel in [("missed_real", lambda o: o["cls"] == "real" and not o["accepted"]),
                      ("accepted_decoy", lambda o: o["cls"] == "decoy" and o["accepted"])]:
        errs = [o for o in r["objects"] if sel(o)]
        print(f"[plot_filter] {kind}: {len(errs)}")
        if errs:
            annotate(by_id[errs[0]["image_id"]], a.raw, labels, out / f"frame_{kind}.png", kind.replace("_", " "))
    print(f"[plot_filter] figures in {out}")


if __name__ == "__main__":
    main()
