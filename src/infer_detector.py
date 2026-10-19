"""
Module 1 - Detection (inference)
==================================
Runs the fine-tuned detector on the evaluation dataset (the custom set
with real + decoy fire extinguishers) and writes out raw detections as
JSON. This file is the CONTRACT between Module 1 (detection) and
Module 2 (decoy filtering): keep the schema stable so your teammate's
code doesn't break.

Output JSON schema (one entry per image):
{
  "image_id": "frame_0001",
  "image_path": "data/eval/images/frame_0001.jpg",
  "detections": [
    {
      "det_id": "frame_0001_0",
      "bbox_xyxy": [x1, y1, x2, y2],     # pixel coords, int
      "confidence": 0.87,
      "class": "fire_extinguisher"
    },
    ...
  ]
}

Usage:
    python src/infer_detector.py \
        --weights runs/detect/fire_extinguisher_yolov8s/weights/best.pt \
        --source data/eval/images \
        --conf 0.25 \
        --out outputs/detections.json
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run the fine-tuned detector on evaluation images")
    p.add_argument("--weights", type=str, required=True,
                    help="Path to trained weights (best.pt from train_detector.py)")
    p.add_argument("--source", type=str, required=True,
                    help="Folder of images to run detection on (the eval/custom dataset)")
    p.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold. Report this exact value "
                         "in your slides - the assignment asks for it explicitly.")
    p.add_argument("--iou", type=float, default=0.45,
                    help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=str, default="outputs/detections.json")
    p.add_argument("--save_vis", action="store_true",
                    help="Also save annotated images with boxes drawn (for your 'Visual Evidence' slide)")
    p.add_argument("--vis_dir", type=str, default="outputs/vis_detections")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    source_dir = Path(args.source)
    image_paths = sorted(
        [p for p in source_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    print(f"[infer_detector] Running on {len(image_paths)} images, conf={args.conf}, iou={args.iou}")

    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save_vis,
        project=str(Path(args.vis_dir).parent) if args.save_vis else None,
        name=Path(args.vis_dir).name if args.save_vis else None,
        exist_ok=True,
        verbose=False,
    )

    output = []
    for img_path, result in zip(image_paths, results):
        image_id = img_path.stem
        detections = []
        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = model.names[cls_id]
            detections.append({
                "det_id": f"{image_id}_{i}",
                "bbox_xyxy": [round(v, 1) for v in xyxy],
                "confidence": round(conf, 4),
                "class": cls_name,
            })
        output.append({
            "image_id": image_id,
            "image_path": str(img_path),
            "detections": detections,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    total_dets = sum(len(entry["detections"]) for entry in output)
    print(f"[infer_detector] Wrote {total_dets} detections across {len(output)} images -> {out_path}")
    if args.save_vis:
        print(f"[infer_detector] Annotated images saved under: {args.vis_dir}")


if __name__ == "__main__":
    main()
