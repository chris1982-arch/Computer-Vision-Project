# R7020E Project 1 - Fire-extinguisher detection, decoy rejection, 3D localization

Team of 3, one module each:
- Module 1 - detection (done), Chris Palantzidis: `src/train_detector.py`, `src/infer_detector.py`, `src/infer_colab.py`
- Module 2 - real vs decoy filtering, Rafal Popiolek: `src/decoy_filter.py` (TODO)
- Module 3 - 3D localization, baseline and evaluation, Vaishnavi Dasari: `src/localize_3d.py`, `src/evaluate.py` (TODO)

## Data contract (each module reads the previous JSON and adds fields)
1. `outputs/detections_{eval,test}.json` - Module 1: `image_id`, `image_path`, `detections[{det_id, bbox_xyxy, confidence, class}]`
2. `outputs/filtered.json` - Module 2 adds `is_real` (bool) and `decoy_score` (float) per detection
3. `outputs/localized.json` - Module 3 adds `position_3d {x, y, z, unit: "m", frame: "camera"}` for accepted real detections

Image ids are the timestamp in the file name (`camera_color_image_<ns>.png`); depth frames are matched to colour frames by nearest timestamp (117 depth frames for 215 colour frames, so record the time gap and reject matches that are too far apart).

## Labels (reference boxes, real vs decoy)
Drawn by hand in Roboflow, YOLO format, one `.txt` per image: `class x_center y_center width height` (normalized 0-1).
- Class ids: **0 = decoy, 1 = real** (always map by class name from `data.yaml`, not by number; ids depend on the Roboflow project)
- File names end with `.rf.<hash>` (added by Roboflow); the image name is the part before `.rf.`
- `labels/chris_test/` - test recording, 88 frames (Chris)
- eval recording, 127 frames: first 64 frames (Rafal), last 63 frames (Vaishnavi) - to be added under `labels/`

## Evaluation protocol
- `test` recording = development set (our choice; tune thresholds here); `eval` recording = final measurement (run once)
- A detection matches an annotated object when the centre of the predicted box lies inside the reference box
- Real-object TPR = real objects detected and accepted as real / all real objects (target 0.75 or more)
- Decoy FPR = decoys detected and accepted as real / all decoys (target 0.15 or less)
- Baseline: detector without decoy filtering (every detection is real); Z from the median depth in the box, x and y from the box centre pixel and the camera intrinsics; same data and metrics
- Detector confidence threshold 0.25; decoy-rejection threshold: [to be filled by Module 2]
- 3D position: camera frame, metres

## Module 1 - detector
- Model: YOLO26s (Ultralytics), COCO-pretrained, fine-tuned on the Roboflow fire-extinguisher set (1 class `extinguisher`; 2934 train and 328 validation images, no Roboflow augmentation; Ultralytics training augmentations were used)
- Validation (328 images, 606 instances): P 0.927, R 0.901, mAP50 0.948, mAP50-95 0.790
- Comparison model: YOLOv8s (mAP50 0.947, mAP50-95 0.771)
- Training: 50 epochs max, imgsz 640, batch 16, patience 15, Tesla T4
- Inference settings: conf 0.25, NMS iou 0.45, imgsz 640
- Recordings: `eval` = 127 colour frames, `test` = 88 colour frames (same file layout as the course data)
- Output on our data: eval 301 detections, test 150 detections (before decoy filtering)
- Known limitation: sometimes two boxes on one real extinguisher (horn+body vs body only); to be merged in Module 2
- Validation numbers are on the public dataset, not on our recordings. Leakage check: no source image name and no identical file is in both train and validation; the 328 validation files come from 207 source names

## Reproduce
Training (final model; the Roboflow set exported as YOLOv8 format, `data/fire_extinguisher.yaml` pointing to its train/valid folders):
```
python src/train_detector.py --data data/fire_extinguisher.yaml --model yolo26s.pt --epochs 50 --imgsz 640 --batch 16 --patience 15 --name fire_ext_26s
```
Inference (only the colour frames `camera_color_image_*.png` are read; depth/IR files are ignored):
```
pip install -r requirements.txt
# put weights in models/best.pt and the colour frames in data/eval and data/test
python src/infer_colab.py
```
Software: ultralytics 8.4.182, Python 3.13.15 (Colab), torch 2.11.0+cu130, GPU Tesla T4.

## Data (not in the repo)
Training set: Roboflow "FireExtinguisher" version 1 (YOLOv8 export). Eval/test recordings: course OneDrive link (rosbag + raw sensor folders).
