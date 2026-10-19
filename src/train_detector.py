"""
Module 1 - Detection
=====================
Fine-tunes a pretrained YOLOv8 model on the fire extinguisher training
dataset. This is transfer learning, not training from scratch: we start
from COCO-pretrained weights and adapt them to our single class.

Why this design (for your slide deck / oral exam):
- YOLOv8 is a single-stage detector -> fast enough for "real-time" as the
  assignment asks for.
- Starting from pretrained weights is explicitly allowed by the assignment
  ("existing pretrained models can be used with clarity on their source
  and role"). Source: Ultralytics YOLOv8, pretrained on COCO.
- We pick the 's' (small) variant by default: a reasonable accuracy/speed
  trade-off for a laptop/Colab GPU, good for the "real-time" requirement.
  Swap to 'n' (nano) if training/inference is too slow, or 'm' (medium)
  if you have GPU budget and want higher accuracy.

Usage:
    python src/train_detector.py --data data/fire_extinguisher.yaml \
        --model yolo26s.pt --epochs 50 --imgsz 640 --batch 16

Outputs:
    runs/detect/train*/weights/best.pt   <- use this for inference later
    runs/detect/train*/results.png       <- loss/mAP curves for your slides
    runs/detect/train*/confusion_matrix.png
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the fire extinguisher dataset")
    p.add_argument("--data", type=str, default="data/fire_extinguisher.yaml",
                    help="Path to the YOLO data config (train/val paths + class names)")
    p.add_argument("--model", type=str, default="yolo26s.pt",
                    help="Pretrained checkpoint to start from (yolov8n/s/m/l/x.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=15,
                    help="Early-stopping patience (epochs with no val improvement)")
    p.add_argument("--device", type=str, default="0",
                    help="'0' for first GPU, 'cpu' if no GPU available")
    p.add_argument("--project", type=str, default="runs/detect",
                    help="Where Ultralytics writes run folders")
    p.add_argument("--name", type=str, default="fire_ext_26s",
                    help="Run name -> runs/detect/<name>/")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data config not found at {data_path}. "
            f"Download the fire extinguisher dataset from Canvas, arrange it "
            f"in YOLO format (see comments in data/fire_extinguisher.yaml), "
            f"and point --data at the yaml file."
        )

    print(f"[train_detector] Loading pretrained weights: {args.model}")
    model = YOLO(args.model)

    print(f"[train_detector] Starting fine-tuning: "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")

    # NOTE for the report: record these exact hyperparameters (and the
    # Ultralytics version from requirements.txt) in your "Evaluation Setup"
    # slide - the assignment explicitly asks for "parameter settings,
    # software versions".
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,          # saves results.png, confusion_matrix.png, PR curves
        save=True,
        exist_ok=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[train_detector] Done. Best weights saved at: {best_weights}")
    print("[train_detector] Use this path with src/infer_detector.py or src/decoy_filter.py")


if __name__ == "__main__":
    main()
