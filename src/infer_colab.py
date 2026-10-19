"""Module 1 - inference used for our final results (run in Google Colab, T4 GPU).
Runs the detector on the colour frames of the eval and test recordings and writes
detections_eval.json / detections_test.json (same schema as infer_detector.py).
Settings: ultralytics==8.4.182, conf=0.25, iou=0.45, imgsz=640.
"""
import json
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = "models/best.pt"          # YOLO26s fine-tuned on the fire-extinguisher dataset
DATA = "data"                        # contains eval/ and test/ folders with colour PNGs
OUT = "outputs"
CONF, IOU, IMGSZ = 0.25, 0.45, 640

model = YOLO(WEIGHTS)
Path(OUT).mkdir(exist_ok=True)
for split in ["eval", "test"]:
    imgs = sorted(Path(f"{DATA}/{split}").rglob("camera_color_image_*.png"))
    results = model.predict([str(p) for p in imgs], conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
    out = []
    for p, r in zip(imgs, results):
        dets = [{"det_id": f"{p.stem}_{i}",
                 "bbox_xyxy": [round(v, 1) for v in r.boxes.xyxy[i].tolist()],
                 "confidence": round(float(r.boxes.conf[i]), 4),
                 "class": "extinguisher"} for i in range(len(r.boxes))]
        out.append({"image_id": p.stem, "image_path": str(p), "detections": dets})
    json.dump(out, open(f"{OUT}/detections_{split}.json", "w"), indent=1)
    print(split, len(imgs), "images,", sum(len(o["detections"]) for o in out), "detections")
