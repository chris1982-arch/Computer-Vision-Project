# R7020E Project 1 - Fire-extinguisher detection, decoy rejection, 3D localization

Team of 3, one module each:

| Module | Owner | Files |
|---|---|---|
| 1 - detector | Chris Palantzidis | `src/train_detector.py`, `src/infer_detector.py`, `src/infer_colab.py`, `src/count_detections.py`, notebook `Computer_Vision_Project (1).ipynb` |
| 2 - real vs decoy filter | Rafal Popiolek | `src/decoy_filter.py`, `src/labels.py` |
| 3 - 3D localization, baseline, evaluation | Vaishnavi Dasari | `src/localize_3d.py`, `src/evaluate.py`, `src/visualize_results.py`, `src/plot_3d_example.py` |

## Pipeline and data contract
Each module reads the previous JSON and adds fields:
1. `outputs/detections_{eval,test}.json` - Module 1: `image_id`, `image_path`, `detections[{det_id, bbox_xyxy, confidence, class}]`
2. `outputs/filtered_{eval,test}.json` - Module 2 adds `is_real`, `decoy_score` (0 real, 1 decoy), `flatness`, `height_m`, `z_med` (m), `depth_gap_ms` per detection
3. `outputs/localized_{eval,test}.json` - Module 3 adds `position_3d {x, y, z, unit: "m", frame: "camera"}` for accepted real detections
4. `outputs/evaluation_{eval,test}.txt` - Module 3: baseline against final system

Image ids are the timestamp in the file name (`camera_color_image_<ns>.png`). Depth frames are matched to colour frames by the nearest timestamp, at most 100 ms apart (depth frames: test 48 for 88 colour frames, eval 69 for 127).

## Labels (reference boxes, real vs decoy)
Drawn by hand in Roboflow, YOLO format, one `.txt` per image: `class x_center y_center width height` (normalized 0-1).
- Class ids depend on the Roboflow project: always map by class name from each folder's `data.yaml` (`src/labels.py` does this and rejects frames labelled twice).
- `labels/chris_test/` - test recording, 88 frames (Chris)
- `labels/eval_first64/` - eval recording, first 64 frames (Rafal); one box corrected real to decoy after export, noted in its README
- `labels/eval_last63/` - eval recording, last 63 frames (Vaishnavi)

## Evaluation protocol
- `test` recording = development set (thresholds tuned here); `eval` recording = final measurement (run once, thresholds frozen)
- A detection matches an annotated object when the centre of the predicted box lies inside the reference box (`src/labels.py`)
- Real-object TPR = real objects detected and accepted as real / all real objects (target 0.75 or more)
- Decoy FPR = decoys detected and accepted as real / all decoys (target 0.15 or less)
- Baseline: detector without decoy filtering (every detection is real), same data and metrics
- Detector confidence threshold 0.25, NMS IoU 0.45
- 3D position: camera frame (`camera_color_optical_frame`), X right, Y down, Z forward, metres

### Results
Eval recording (127 frames, 127 real, 226 decoys), run once:

| System | TPR | FPR | Accepted real | Missed real | Accepted decoys |
|---|---|---|---|---|---|
| Baseline (no filter) | 0.953 | 0.743 | 121 | 6 | 168 |
| Final system (depth filter) | 0.858 | 0.000 | 109 | 18 | 0 |

Both targets are met by the final system; the baseline fails the FPR target. Of the 18 missed real objects, the detector misses 6 and the filter rejects 12.

Dev recording, test (88 frames, 57 real, 138 decoys):

| System | TPR | FPR | Accepted real | Missed real | Accepted decoys |
|---|---|---|---|---|---|
| Baseline (no filter) | 1.000 | 0.638 | 57 | 0 | 88 |
| Final system (depth filter) | 0.965 | 0.022 | 55 | 2 | 3 |

There is no measured ground-truth 3D position, so the 3D examples are qualitative and no 3D error is reported.

## Module 1 - detector (Chris)
- Model: YOLO26s (Ultralytics), COCO-pretrained, fine-tuned on the Roboflow "FireExtinguisher" v1 set (1 class `extinguisher`); YOLOv8s trained for comparison
- Data: 3262 images = 2934 train + 328 validation (606
