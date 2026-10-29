"""
Evaluation plots and visual examples
Run from project root: python src/visualize_results.py
Requires matplotlib, opencv-python, and the existing src/labels.py.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from labels import load_labels, evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
LABEL_DIRS = [ROOT / "labels/eval_first64", ROOT / "labels/eval_last63"]
IMAGE_DIR = ROOT / "data/raw/eval/camera_color_image_raw"

BLUE = (230, 90, 25)       # ground truth
GREEN = (40, 160, 40)      # accepted prediction
RED = (35, 35, 220)        # rejected prediction


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


def center_inside(box, gt):
    x1, y1, x2, y2 = box
    a, b, c, d = gt
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return a <= cx <= c and b <= cy <= d


def find_image(image_id):
    candidates = list(IMAGE_DIR.glob(f"{image_id}.*"))
    if not candidates:
        candidates = list(IMAGE_DIR.glob(f"*{image_id.split('_')[-1]}*"))
    for path in candidates:
        image = cv2.imread(str(path))
        if image is not None:
            return image, path
    return None, None


def dashed_rectangle(image, box, color, thickness=3, dash=10):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def save_focused_example(image_id, title, caption, objects, detections,
                         focus_cls, target_gt, filename):
    image, source = find_image(image_id)
    if image is None:
        print(f"Could not find source image for {image_id}; skipped {filename}")
        return

    h, w = image.shape[:2]
    sx, sy = w / 640.0, h / 400.0

    x1, y1, x2, y2 = target_gt
    gt_scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    cv2.rectangle(image, (int(gt_scaled[0]), int(gt_scaled[1])),
                  (int(gt_scaled[2]), int(gt_scaled[3])), BLUE, 3)


    relevant = [d for d in detections if center_inside(d["bbox_xyxy"], target_gt)]
    for det in relevant:
        b = det["bbox_xyxy"]
        scaled = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        accepted = det.get("is_real", True)
        dashed_rectangle(image, scaled, GREEN if accepted else RED, thickness=3)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="#195AE6", linewidth=3, linestyle="-", label="Ground-truth box"),
        Line2D([0], [0], color="#28A028", linewidth=3, linestyle="--", label="Accepted prediction"),
        Line2D([0], [0], color="#DC2323", linewidth=3, linestyle="--", label="Rejected prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               fontsize=10, bbox_to_anchor=(0.5, 0.105))
    fig.text(0.5, 0.055, caption, ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f4f7",
                       edgecolor="#c8cdd3"))
    fig.tight_layout(rect=[0.02, 0.20, 0.98, 0.94])
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / filename} (source: {source.name}; relevant predictions: {len(relevant)})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABEL_DIRS)
    baseline = read_json(ROOT / "outputs/detections_eval.json")
    final = read_json(ROOT / "outputs/filtered_eval.json")
    localized = read_json(ROOT / "outputs/localized_eval.json")
    bm, fm = evaluate(baseline, labels), evaluate(final, labels)

    # Performance charts
    chart_specs = [
        ("TPR", [bm["TPR"] * 100, fm["TPR"] * 100], 75, "TPR (%)",
         "Real-object acceptance — evaluation set", "performance_tpr.png", 105),
        ("FPR", [bm["FPR"] * 100, fm["FPR"] * 100], 15, "FPR (%)",
         "Decoy false acceptance — evaluation set", "performance_fpr.png", None),
    ]
    for metric, values, target, ylabel, title, filename, ymax in chart_specs:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        bars = ax.bar(["Baseline", "Final system"], values)
        ax.axhline(target, color="#555555", linestyle="--",
                   label=f"Target: {'≥' if metric == 'TPR' else '≤'} {target}%")
        ax.set_ylim(0, ymax or max(20, max(values) + 10))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                    f"{value:.1f}%", ha="center", va="bottom", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3D estimates
    points = []
    for frame in localized:
        for det in frame.get("detections", []):
            pos = det.get("position_3d")
            if det.get("is_real", False) and pos:
                points.append((pos["x"], pos["y"], pos["z"]))
    if points:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points])
        ax.set_xlabel("X (m) — right")
        ax.set_ylabel("Y (m) — down")
        ax.set_zlabel("Z (m) — forward")
        ax.set_title(f"Estimated 3D positions of accepted detections (n={len(points)})")
        fig.tight_layout()
        fig.savefig(OUT / "localization_3d.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    final_by_id = {f["image_id"]: f.get("detections", []) for f in final}


    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    real_candidates, decoy_candidates, missed_candidates, missed_fallback = [], [], [], []
    for image_id, objects in labels.items():
        detections = final_by_id.get(image_id, [])
        decoy_gts = [gt for cls, gt in objects if cls == "decoy"]
        real_gts = [gt for cls, gt in objects if cls == "real"]



        for cls, gt in objects:
            matched = [d for d in detections if center_inside(d["bbox_xyxy"], gt)]
            if cls == "real":
                accepted = [d for d in matched if d.get("is_real", True)]
                rejected = [d for d in matched if not d.get("is_real", True)]
                if len(accepted) == 1 and not rejected:


                    real_candidates.append((image_id, objects, gt, detections))

        # all decoy GT boxes in the frame must have an explicit

        if decoy_gts:
            all_decoys_accounted_for = True
            total_rejected_matches = 0
            for gt in decoy_gts:
                matched = [d for d in detections if center_inside(d["bbox_xyxy"], gt)]
                rejected = [d for d in matched if not d.get("is_real", True)]
                accepted = [d for d in matched if d.get("is_real", True)]
                if not rejected or accepted:
                    all_decoys_accounted_for = False
                    break
                total_rejected_matches += len(rejected)
            if all_decoys_accounted_for:
                decoy_candidates.append((len(decoy_gts), total_rejected_matches,
                                         image_id, objects, decoy_gts[0], detections))

        # no matching prediction, no overlap with other

        for cls, gt in objects:
            if cls != "real":
                continue
            matched = [d for d in detections if center_inside(d["bbox_xyxy"], gt)]
            if matched:
                continue
            x1, y1, x2, y2 = gt
            margin = min(x1, y1, 640 - x2, 400 - y2)
            other_gts = [other for other_cls, other in objects if other is not gt]
            overlaps_other = any(iou(gt, other) > 0.05 for other in other_gts)


            fallback_score = margin - (1000 if overlaps_other else 0)
            missed_fallback.append((fallback_score, image_id, objects, gt, detections))

            # fully visible box and no overlap with other GT
            if margin >= 20 and not overlaps_other:
                missed_candidates.append((margin, image_id, objects, gt, detections))

    if real_candidates:
        image_id, objects, gt, detections = real_candidates[0]
        save_focused_example(
            image_id,
            "Correct detection and acceptance of a real object",
            "Ground truth: REAL  |  One matching prediction: ACCEPTED",
            objects, detections, "real", gt, "example_real_accepted.png")
    else:
        (OUT / "example_real_accepted.png").unlink(missing_ok=True)
        print("No clean accepted-real example found; removed any stale example_real_accepted.png.")

    if decoy_candidates:
        # the frame with the most decoy GT objects all explicitly rejected.
        _, _, image_id, objects, gt, detections = max(
            decoy_candidates, key=lambda item: (item[0], item[1]))


        image, source = find_image(image_id)
        if image is not None:
            h, w = image.shape[:2]
            sx, sy = w / 640.0, h / 400.0
            for cls, box in objects:
                if cls == "decoy":
                    scaled = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
                    cv2.rectangle(image, (int(scaled[0]), int(scaled[1])),
                                  (int(scaled[2]), int(scaled[3])), BLUE, 3)
            for det in detections:
                if det.get("is_real", True):
                    continue
                if any(cls == "decoy" and center_inside(det["bbox_xyxy"], gt_box)
                       for cls, gt_box in objects):
                    b = det["bbox_xyxy"]
                    scaled = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
                    dashed_rectangle(image, scaled, RED, thickness=3)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(rgb)
            ax.set_title("Correct rejection of printed/displayed decoys", fontsize=16,
                         fontweight="bold", pad=14)
            ax.axis("off")
            handles = [
                Line2D([0], [0], color="#195AE6", linewidth=3, linestyle="-",
                       label="Ground-truth decoy"),
                Line2D([0], [0], color="#DC2323", linewidth=3, linestyle="--",
                       label="Rejected prediction"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
                       fontsize=10, bbox_to_anchor=(0.5, 0.105))
            fig.text(0.5, 0.055,
                     f"All {len([1 for cls, _ in objects if cls == 'decoy'])} decoy ground-truth boxes in this frame have rejected matching predictions.",
                     ha="center", va="center", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f4f7",
                               edgecolor="#c8cdd3"))
            fig.tight_layout(rect=[0.02, 0.20, 0.98, 0.94])
            fig.savefig(OUT / "example_decoy_rejected.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {OUT / 'example_decoy_rejected.png'} (source: {source.name})")
    else:
        (OUT / "example_decoy_rejected.png").unlink(missing_ok=True)
        print("No frame found where every decoy GT has a rejected match and none is accepted.")
        print("Removed any stale example_decoy_rejected.png; do not use an old image as if it were newly selected.")

    if missed_candidates:
        _, image_id, objects, gt, detections = max(missed_candidates, key=lambda item: item[0])
        caption = "Ground truth: REAL  |  No matching detector prediction"
        print("Selected a clean missed-real example (no GT overlap and adequate image margin).")
    elif missed_fallback:
        _, image_id, objects, gt, detections = max(missed_fallback, key=lambda item: item[0])
        caption = "Ground truth: REAL  |  No matching detector prediction (best available sample)"
        print("No fully clean missed-real sample was available; selected the best available no-match sample.")
    else:
        image_id = objects = gt = detections = None

    if image_id is not None:
        save_focused_example(
            image_id,
            "Missed real object (no matching prediction)",
            caption,
            objects, detections, "real", gt, "example_real_missed.png")
    else:
        print("No missed-real candidate exists in the evaluation labels; no third image generated.")

    print("\nEvaluation-set counts:")
    print(f"Real accepted: {fm['real_accepted']}/{fm['n_real']}; missed/rejected: {fm['n_real'] - fm['real_accepted']}")
    print(f"Decoys accepted: {fm['decoy_accepted']}/{fm['n_decoy']}")
    print(f"3D positions plotted: {len(points)}")
    print("Decoys incorrectly accepted: none observed in evaluation set." if fm["decoy_accepted"] == 0 else
          f"Decoys incorrectly accepted: {fm['decoy_accepted']}")
    print(f"Plots saved under: {OUT}")


if __name__ == "__main__":
    main()
