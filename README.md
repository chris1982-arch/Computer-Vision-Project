R7020E Project 1 - Fire-extinguisher detection, decoy rejection, 3D localization
Team of 3, one module each:
Module	Owner	Files
1 - detector	Chris Palantzidis	`src/train_detector.py`, `src/infer_detector.py`, `src/infer_colab.py`, `src/count_detections.py`, notebook `Computer_Vision_Project (1).ipynb`
2 - real vs decoy filter	Rafal Popiolek	`src/decoy_filter.py`, `src/labels.py`
3 - 3D localization, baseline, evaluation	Vaishnavi Dasari	`src/localize_3d.py`, `src/evaluate.py`, `src/visualize_results.py`, `src/plot_3d_example.py`
Pipeline and data contract
Each module reads the previous JSON and adds fields:
`outputs/detections_{eval,test}.json` - Module 1: `image_id`, `image_path`, `detections[{det_id, bbox_xyxy, confidence, class}]`
`outputs/filtered_{eval,test}.json` - Module 2 adds `is_real`, `decoy_score` (0 real, 1 decoy), `flatness`, `height_m`, `z_med` (m), `depth_gap_ms` per detection
`outputs/localized_{eval,test}.json` - Module 3 adds `position_3d {x, y, z, unit: "m", frame: "camera"}` for accepted real detections
`outputs/evaluation_{eval,test}.txt` - Module 3: baseline against final system
Image ids are the timestamp in the file name (`camera_color_image_<ns>.png`). Depth frames are matched to colour frames by the nearest timestamp, at most 100 ms apart (depth frames: test 48 for 88 colour frames, eval 69 for 127).
Labels (reference boxes, real vs decoy)
Drawn by hand in Roboflow, YOLO format, one `.txt` per image: `class x_center y_center width height` (normalized 0-1).
Class ids depend on the Roboflow project: always map by class name from each folder's `data.yaml` (`src/labels.py` does this and rejects frames labelled twice).
`labels/chris_test/` - test recording, 88 frames (Chris)
`labels/eval_first64/` - eval recording, first 64 frames (Rafal): 64 real, 109 decoys; one box corrected real to decoy after export, noted in its README
`labels/eval_last63/` - eval recording, last 63 frames (Vaishnavi)
Evaluation protocol
`test` recording = development set (thresholds tuned here); `eval` recording = final measurement (run once, thresholds frozen)
A detection matches an annotated object when the centre of the predicted box lies inside the reference box (`src/labels.py`)
Real-object TPR = real objects detected and accepted as real / all real objects (target 0.75 or more)
Decoy FPR = decoys detected and accepted as real / all decoys (target 0.15 or less)
Baseline: detector without decoy filtering (every detection is real), same data and metrics
Detector confidence threshold 0.25, NMS IoU 0.45
3D position: camera frame (`camera_color_optical_frame`), X right, Y down, Z forward, metres
Results
Eval recording (127 frames, 127 real, 226 decoys), run once:
System	TPR	FPR	Accepted real	Missed real	Accepted decoys
Baseline (no filter)	0.953	0.743	121	6	168
Final system (depth filter)	0.858	0.000	109	18	0
Both targets are met by the final system; the baseline fails the FPR target. Of the 18 missed real objects, the detector misses 6 and the filter rejects 12.
Dev recording, test (88 frames, 57 real, 138 decoys):
System	TPR	FPR	Accepted real	Missed real	Accepted decoys
Baseline (no filter)	1.000	0.638	57	0	88
Final system (depth filter)	0.965	0.022	55	2	3
There is no measured ground-truth 3D position, so the 3D examples are qualitative and no 3D error is reported.
Module 1 - detector (Chris)
Model: YOLO26s (Ultralytics), COCO-pretrained, fine-tuned on the Roboflow "FireExtinguisher" v1 set (1 class `extinguisher`); YOLOv8s trained for comparison
Data: 3262 images = 2934 train + 328 validation (606 instances), Roboflow export, no Roboflow augmentation (Ultralytics training augmentations were used, see the run's `args.yaml`); no shared source names or identical files between train and validation (the 328 validation files come from 207 source names)
Validation (public set, not our recordings): P 0.927, R 0.901, mAP50 0.948, mAP50-95 0.790 (YOLOv8s: mAP50 0.947, mAP50-95 0.771)
Training: 50 epochs, imgsz 640, batch 16, patience 15 (all 50 epochs ran), Tesla T4, about 51 minutes
Inference: conf 0.25, NMS iou 0.45, imgsz 640; on Tesla T4 7.5 ms inference per image, 49.3 ms per image including image loading (127 eval frames in 6.3 s)
Output on our data: eval 127 frames, 301 detections; test 88 frames, 150 detections (before decoy filtering; 299 and 148 after duplicate merging); counts checked with `src/count_detections.py`
Known limitation: sometimes two boxes on one real extinguisher (horn plus body, body only); duplicate boxes are merged in Module 2
Module 2 - real vs decoy filter (Rafal)
A printed decoy is flat and small; a real extinguisher (cylinder and hose) is not flat and about 1 m tall in its box. Depth is registered to the colour camera, so each detection box is measured directly in the depth image.
Duplicate boxes: a box more than 80 % inside a more confident box is dropped
Depth frame: nearest by timestamp, at most 100 ms away
Per box, from the depth image inside the box (15 % left/right and 10 % top/bottom trimmed): plane-fit flatness (RMS in mm / z squared, z in m) and physical box height in m
Rule: a detection is real if flatness > 5 and height > 0.7 m (`FLAT_MIN`, `HEIGHT_MIN` in `src/decoy_filter.py`); thresholds chosen on the test recording, then frozen
Without usable depth (no depth frame within 100 ms, or too few valid pixels): accepted only when detector confidence is at least 0.85 (eval: 4 detections without usable depth, none accepted)
Module 3 - 3D localization and evaluation (Vaishnavi)
`src/localize_3d.py`: 3D position of accepted real detections Z = median of the valid depth pixels in an 11 x 11 patch at the box centre (at least 5 valid pixels, otherwise no position); X, Y from the box centre pixel and the camera intrinsics (`camera_info`); camera frame, metres
`src/evaluate.py`: baseline (every detection real) against the final system, TPR and FPR with the metric above
`src/visualize_results.py`, `src/plot_3d_example.py`: result figures and example frames
Reproduce
Environment (Python 3 with the packages in `requirements.txt`):
```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```
(On Windows use `.venv\Scripts\python`; or skip the venv and add `PY=python` to every `make` command.)
Data (not in the repo): the course recordings, unpacked to `data/raw/test/` and `data/raw/eval/`, each with the colour PNGs (`camera_color_image_<ns>.png`) and the sub-folders `camera_depth_image_raw/` (depth PNGs, uint16 mm) and `camera_color_camera_info/` (text file with the camera matrix K).
Modules 2 and 3 run from the saved Module 1 output in `outputs/detections_*.json` (no GPU needed):
```
make module2 score localize evaluate figures
```
`make module2`: filter test (with TPR/FPR) and eval
`make score`: final eval TPR/FPR
`make localize`: 3D positions
`make evaluate`: baseline against final system -> `outputs/evaluation_{test,eval}.txt`
`make figures`: plots and example frames -> `outputs/figures/`
Module 1 (only needed to recreate `outputs/detections_*.json`; the Roboflow set exported as YOLOv8 format, `data/fire_extinguisher.yaml` pointing to its train/valid folders):
```
python src/train_detector.py --data data/fire_extinguisher.yaml --model yolo26s.pt --epochs 50 --imgsz 640 --batch 16 --patience 15 --name fire_ext_26s
# weights are in models/best.pt; infer_colab.py reads colour PNGs from data/eval and data/test
# (change DATA = "data" to DATA = "data/raw" in the file to use the layout above)
python src/infer_colab.py
python src/count_detections.py
```
Software (Module 1): ultralytics 8.4.182, Python 3.13 (Colab), torch 2.11 (CUDA), GPU Tesla T4.
Contribution record
Chris: Module 1 (data, training, model comparison, inference, JSON, repo), labels for the test recording
Rafal: Module 2 (decoy filter, duplicate merging), labels for the first 64 eval frames, Makefile, demo video
Vaishnavi: Module 3 (3D localization, baseline, evaluation, figures), labels for the last 63 eval frames
Slides: each member wrote and presents the slides of their own module
Data (not in the repo)
Training set: Roboflow "FireExtinguisher v1" (YOLOv8 export). Eval/test recordings: course OneDrive link (rosbag and raw sensor folders). Figures in `outputs/figures/` are generated, not stored.
