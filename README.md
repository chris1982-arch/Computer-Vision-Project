# R7020E Project 1 - Fire-extinguisher detection, decoy rejection, 3D localization

Team of 3, one module each:
- Module 1 - detection (done), Chris Palantzidis: `src/train_detector.py`, `src/infer_detector.py`, `src/infer_colab.py`
- Module 2 - real vs decoy filtering (done, eval score pending full eval labels), Rafal Popiolek: `src/decoy_filter.py`, `src/depth_utils.py`, `src/labels.py`, `src/plot_filter.py`
- Module 3 - 3D localization, baseline and evaluation, Vaishnavi Dasari: `src/localize_3d.py`, `src/evaluate.py` (TODO)

## Data contract (each module reads the previous JSON and adds fields)
1. `outputs/detections_{eval,test}.json` - Module 1: `image_id`, `image_path`, `detections[{det_id, bbox_xyxy, confidence, class}]`
2. `outputs/filtered_{eval,test}.json` - Module 2: `{"meta": {...settings...}, "images": [...]}`; images keep the Module 1 schema (duplicate boxes merged) and each detection gets `is_real` (bool), `decoy_score` (float, 1 - P(real)), `decision` (`depth` or `confidence_fallback`), `depth_ok`, `depth_frame`, `depth_gap_ms`, `merged_from`, `features{...}` (incl. `z_med` in metres). `labels.load_images()` reads both the list and the dict form
3. `outputs/localized.json` - Module 3 adds `position_3d {x, y, z, unit: "m", frame: "camera"}` for accepted real detections

Image ids are the timestamp in the file name (`camera_color_image_<ns>.png`); depth frames are matched to colour frames by nearest timestamp (117 depth frames for 215 colour frames, so record the time gap and reject matches that are too far apart).

## Labels (reference boxes, real vs decoy)
Drawn by hand in Roboflow, YOLO format, one `.txt` per image: `class x_center y_center width height` (normalized 0-1).
- Class ids: **0 = decoy, 1 = real** (always map by class name from `data.yaml`, not by number; ids depend on the Roboflow project)
- File names end with `.rf.<hash>` (added by Roboflow); the image name is the part before `.rf.`
- `labels/chris_test/` - test recording, 88 frames (Chris)
- `labels/eval_last63/` - eval recording, last 63 frames (Vaishnavi): 63 real, 117 decoy
- eval recording, first 64 frames (Rafal) - TODO, goes to `labels/eval_first64/`
- `labels/chris_test/` counts: 57 real, 138 decoy

## Evaluation protocol
- `test` recording = development set (our choice; tune thresholds here); `eval` recording = final measurement (run once)
- A detection matches an annotated object when the centre of the predicted box lies inside the reference box
- Real-object TPR = real objects detected and accepted as real / all real objects (target 0.75 or more)
- Decoy FPR = decoys detected and accepted as real / all decoys (target 0.15 or less)
- Baseline: detector without decoy filtering (every detection is real); Z from the median depth in the box, x and y from the box centre pixel and the camera intrinsics; same data and metrics
- Detector confidence threshold 0.25; decoy-rejection threshold: accept as real when P(real) >= 0.5 (`decoy_score` <= 0.5); without usable depth: confidence >= 0.85
- Metric code: `src/labels.py` (`evaluate()`), shared by Module 2 tuning and Module 3 evaluation. Baseline check: `python src/labels.py --labels labels/chris_test --dets outputs/detections_test.json` -> TPR 1.000 (57/57), FPR 0.638 (88/138)
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

## Module 2 - real vs decoy filter
Idea: a print is flat and small; a real extinguisher is a cylinder with a hose and handle, about 1 m tall in its box.
1. Duplicate boxes in one frame are merged (IoU > 0.5 or intersection / smaller area > 0.8), keeping the higher-confidence box (test: 2 merges, eval: 3).
2. Depth: the depth images are already registered to the colour camera (same K, frame `camera_color_optical_frame`), uint16 millimetres. The nearest depth frame by timestamp is used if the gap is at most 100 ms (test max 87 ms; eval: 2 frames above 100 ms).
3. Features on the inner box (15 % trimmed left/right, 10 % top/bottom; depth 0.2-8 m): `plane_rms_norm` = RMS distance of the back-projected points to their least-squares plane (one 2.5-sigma trim) in mm, divided by z^2; `height_m` = box height in pixels * median z / fy. Also computed but not used by the decision: bulge, depth IQR, step to a background ring, width, valid fraction.
4. Decision: logistic regression (scikit-learn, standardised features) on `log(plane_rms_norm)` and `height_m`, fitted on the test recording (146 detections whose centre lies in one labelled box). Real when P(real) >= 0.5. Parameters in `models/decoy_filter.json` (plain numbers). Without usable depth (no frame within 100 ms or < 30 % valid pixels) the rule is confidence >= 0.85.
5. Feature extraction about 6-7 ms per frame on a laptop CPU.

Dev results (test recording, 88 frames, 57 real, 138 decoys):

| System | TPR (real) | FPR (decoy) | Accepted real | Missed real | Accepted decoys |
|---|---|---|---|---|---|
| Baseline (no filter) | 1.000 | 0.638 | 57 | 0 | 88 |
| Filter, 5-block CV (contiguous frame blocks) | 0.947 | 0.000 | 54 | 3 | 0 |
| Filter, fitted on all dev frames | 0.982 | 0.000 | 56 | 1 | 0 |

Single-feature check on dev: flatness alone (threshold 7.9) gives TPR 0.965 / FPR 0.065; height alone (0.89 m) gives 0.965 / 0.022.
Known failure: when the camera turns fast, the matched depth frame (65 ms away) shows the object somewhere else, so the box measures flat floor and the real extinguisher is rejected (`outputs/figures/frame_missed_real.png`).

Reproduce (Python 3.12, versions in `requirements.txt`; recordings extracted to `data/raw/{eval,test}/`, e.g. `unzip OneDrive_*.zip 'raw/*/camera_color_*' 'raw/*/camera_depth_*' -d data`):
```
python src/decoy_filter.py --split test --fit labels/chris_test --labels labels/chris_test   # fit + dev metrics
python src/decoy_filter.py --split eval --labels labels/eval_first64 labels/eval_last63      # final, run once
python src/plot_filter.py                                                                      # figures -> outputs/figures/
```

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
