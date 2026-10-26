"""Reference labels and the course metric (shared by Module 2 tuning and Module 3 evaluation).

Matching rule (course definition): a detection is associated with an annotated
object when the centre of the predicted box lies inside the reference box.
- real TP: a real object with at least one associated detection accepted as real
- real FN: a real object with no associated detection, or all of them rejected
- decoy FP: a decoy with at least one associated detection accepted as real
TPR = TP / #real objects, FPR = decoy FP / #decoys (all annotated objects count,
also the ones the detector missed).

Usage:
    python src/labels.py --labels labels/chris_test --dets outputs/detections_test.json
    (no is_real field in the JSON = baseline, every detection accepted)
"""
import argparse
import json
from pathlib import Path

import yaml

DEFAULT_NAMES = ["decoy", "real"]


def load_labels(label_dirs, W=640, H=400):
    """{image_id: [{'cls': 'real'|'decoy', 'box': [x1, y1, x2, y2]}]} from YOLO txt files."""
    if isinstance(label_dirs, (str, Path)):
        label_dirs = [label_dirs]
    out = {}
    for d in map(Path, label_dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"label folder {d} not found")
        yml = d / "data.yaml"
        names = yaml.safe_load(yml.read_text())["names"] if yml.exists() else DEFAULT_NAMES
        for f in sorted(d.glob("*.txt")):
            if not f.name.startswith("camera_color_image_"):
                continue
            image_id = f.name.split("_png.rf.")[0]
            objs = []
            for line in f.read_text().split("\n"):
                if not line.strip():
                    continue
                c, xc, yc, w, h = line.split()
                xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
                objs.append({"cls": names[int(c)],
                             "box": [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]})
            if image_id in out:
                raise ValueError(f"{image_id} labelled twice")
            out[image_id] = objs
    return out


def centre_in(det_box, ref_box):
    cx, cy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    return ref_box[0] <= cx <= ref_box[2] and ref_box[1] <= cy <= ref_box[3]


def evaluate(images, labels):
    """images: Module 1/2 JSON list. Detections without 'is_real' count as accepted.

    Returns counts, rates and a per-object list (for figures / error analysis).
    Only frames that have labels are evaluated.
    """
    by_id = {im["image_id"]: im["detections"] for im in images}
    missing = [k for k in labels if k not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} labelled frames have no detections entry, e.g. {missing[0]}")
    objects = []
    for image_id, objs in labels.items():
        dets = by_id[image_id]
        for o in objs:
            assoc = [d for d in dets if centre_in(d["bbox_xyxy"], o["box"])]
            accepted = [d for d in assoc if d.get("is_real", True)]
            objects.append({"image_id": image_id, "cls": o["cls"], "box": o["box"],
                            "detected": bool(assoc), "accepted": bool(accepted),
                            "det_ids": [d["det_id"] for d in assoc]})
    real = [o for o in objects if o["cls"] == "real"]
    decoy = [o for o in objects if o["cls"] == "decoy"]
    tp = sum(o["accepted"] for o in real)
    fp = sum(o["accepted"] for o in decoy)
    return {
        "frames": len(labels), "n_real": len(real), "n_decoy": len(decoy),
        "real_accepted": tp, "real_missed_by_detector": sum(not o["detected"] for o in real),
        "real_rejected": sum(o["detected"] and not o["accepted"] for o in real),
        "decoy_detected": sum(o["detected"] for o in decoy), "decoy_accepted": fp,
        "TPR": tp / len(real) if real else float("nan"),
        "FPR": fp / len(decoy) if decoy else float("nan"),
        "objects": objects,
    }


def load_images(path):
    data = json.load(open(path))
    return data["images"] if isinstance(data, dict) else data


def main():
    p = argparse.ArgumentParser(description="Real-object TPR / decoy FPR with centre-in-box matching")
    p.add_argument("--labels", nargs="+", required=True, help="one or more label folders")
    p.add_argument("--dets", required=True, help="detections_*.json or filtered_*.json")
    a = p.parse_args()
    r = evaluate(load_images(a.dets), load_labels(a.labels))
    r.pop("objects")
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
