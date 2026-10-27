"""Reference labels and the course metric (centre of the detected box inside the reference box).

TPR = real objects with an associated detection accepted as real / all real objects
FPR = decoys with an associated detection accepted as real / all decoys
Detections without "is_real" count as accepted (= baseline).

Usage: python src/labels.py --labels labels/chris_test --dets outputs/detections_test.json
"""
import argparse
import json
from pathlib import Path

import yaml

NAMES = ["decoy", "real"]   # class ids when a label folder has no data.yaml


def load_labels(dirs, W=640, H=400):
    """{image_id: [(cls, [x1, y1, x2, y2])]} from YOLO txt files."""
    out = {}
    for d in map(Path, dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"label folder {d} not found")
        names = NAMES
        if (d / "data.yaml").exists():   # map ids by class name: other Roboflow projects may differ
            names = yaml.safe_load((d / "data.yaml").read_text())["names"]
            names = [{"real": "real", "decoy": "decoy"}.get(str(n).lower()) for n in names]
            if None in names:
                raise ValueError(f"{d}/data.yaml: class names must be 'real' and 'decoy'")
        for f in d.glob("camera_color_image_*.txt"):
            image_id = f.name.split("_png.rf.")[0]
            if image_id in out:
                raise ValueError(f"{image_id} labelled twice (augmented export or overlapping folders?)")
            objs = []
            for c, xc, yc, w, h in (line.split() for line in f.read_text().splitlines() if line.strip()):
                xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
                objs.append((names[int(c)], [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]))
            out[image_id] = objs
    return out


def evaluate(images, labels):
    dets = {im["image_id"]: im["detections"] for im in images}
    n = {"real": 0, "decoy": 0}
    acc = {"real": 0, "decoy": 0}
    for image_id, objs in labels.items():
        for cls, (x1, y1, x2, y2) in objs:
            n[cls] += 1
            acc[cls] += any(x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2
                            and d.get("is_real", True)
                            for d in dets[image_id] for b in [d["bbox_xyxy"]])
    return {"n_real": n["real"], "n_decoy": n["decoy"], "real_accepted": acc["real"],
            "decoy_accepted": acc["decoy"], "TPR": acc["real"] / n["real"], "FPR": acc["decoy"] / n["decoy"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--dets", required=True)
    a = p.parse_args()
    print(evaluate(json.load(open(a.dets)), load_labels(a.labels)))
