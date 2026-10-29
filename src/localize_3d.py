"""
Module 3 - 3D localization
Estimate camera-frame 3D positions for accepted real detections."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TIMESTAMP_GAP_MS = 100.0
PATCH_SIZE = 11
MIN_VALID_DEPTH_PIXELS = 5
DEPTH_SCALE = 0.001
COORDINATE_FRAME = "camera_color_optical_frame"


def extract_timestamp(value):

    matches = re.findall(r"\d{15,}", str(value))
    return int(matches[-1]) if matches else None


def load_camera_intrinsics(info_dir):
    #Read the 3x3 camera matrix K from a camera-info file
    if not info_dir.is_dir():
        raise FileNotFoundError(f"Camera info folder not found: {info_dir}")

    files = sorted(info_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No camera info .txt file found in {info_dir}")

    content = files[0].read_text(encoding="utf-8", errors="replace")
    match = re.search(r"K:\s*\[(.*?)\]", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find camera matrix 'K: [...]' in {files[0]}")

    values = [float(value.strip()) for value in match.group(1).split(",")]
    if len(values) != 9:
        raise ValueError(f"Expected 9 values in camera matrix K; got {len(values)}")

    K = np.asarray(values, dtype=float).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal lengths in camera matrix: fx={fx}, fy={fy}")

    return fx, fy, cx, cy


def index_depth_images(depth_dir):
    #Return timestamp-to-path mapping for depth PNG files
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth image folder not found: {depth_dir}")

    index = {}
    for path in depth_dir.glob("*.png"):
        timestamp = extract_timestamp(path.stem)
        if timestamp is not None:
            index[timestamp] = path

    if not index:
        raise RuntimeError(f"No timestamped depth PNGs found in {depth_dir}")

    return index


def find_nearest_depth(color_timestamp, depth_index):
    #Find nearest depth image and the absolute timestamp gap in milliseconds
    nearest_timestamp = min(
        depth_index,
        key=lambda timestamp: abs(timestamp - color_timestamp),
    )
    gap_ms = abs(nearest_timestamp - color_timestamp) / 1_000_000.0
    return depth_index[nearest_timestamp], gap_ms


def get_median_depth_m(depth_image, center_x, center_y):
    #Estimate depth from the median of valid pixels in a 11x11 centre patch
    if depth_image is None or depth_image.ndim != 2:
        return None

    height, width = depth_image.shape
    x, y = int(round(center_x)), int(round(center_y))
    if not (0 <= x < width and 0 <= y < height):
        return None

    half = PATCH_SIZE // 2
    patch = depth_image[
        max(0, y - half):min(height, y + half + 1),
        max(0, x - half):min(width, x + half + 1),
    ]

    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < MIN_VALID_DEPTH_PIXELS:
        return None

    depth_m = float(np.median(valid)) * DEPTH_SCALE
    if not np.isfinite(depth_m) or depth_m <= 0:
        return None
    return depth_m


def calculate_3d_position(pixel_x, pixel_y, depth_m, fx, fy, cx, cy):
    #Project pixel coordinates and depth into the camera optical frame
    return {
        "x": float((pixel_x - cx) * depth_m / fx),
        "y": float((pixel_y - cy) * depth_m / fy),
        "z": float(depth_m),
        "unit": "m",
        "frame": COORDINATE_FRAME,
    }


def localize_detection(detection, depth_image, intrinsics):
    #Return a 3D position for a detection, or None if depth is unavailable
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    center_x = (float(x1) + float(x2)) / 2.0
    center_y = (float(y1) + float(y2)) / 2.0

    depth_m = get_median_depth_m(depth_image, center_x, center_y)
    if depth_m is None:
        return None

    fx, fy, cx, cy = intrinsics
    return calculate_3d_position(center_x, center_y, depth_m, fx, fy, cx, cy)


def run(split):
    raw_dir = PROJECT_ROOT / "data" / "raw" / split
    input_path = PROJECT_ROOT / "outputs" / f"filtered_{split}.json"
    output_path = PROJECT_ROOT / "outputs" / f"localized_{split}.json"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Filtered detections not found: {input_path}\n"
            "Run Module 2 first to create filtered_{split}.json."
        )

    intrinsics = load_camera_intrinsics(
        raw_dir / "camera_color_camera_info"
    )
    depth_index = index_depth_images(raw_dir / "camera_depth_image_raw")

    with input_path.open("r", encoding="utf-8") as file:
        entries = json.load(file)

    depth_cache = {}
    total_detections = 0
    accepted_real = 0
    localized_count = 0
    no_position_count = 0
    rejected_gap_count = 0
    missing_timestamp_count = 0
    rejected_decoy_count = 0

    for entry in entries:
        detections = entry.get("detections", [])
        total_detections += len(detections)


        color_timestamp = extract_timestamp(entry.get("image_id", ""))
        if color_timestamp is None:
            color_timestamp = extract_timestamp(entry.get("image_path", ""))

        depth_image = None
        gap_ms = None
        matched_depth_path = None

        if color_timestamp is None:
            missing_timestamp_count += 1
        else:
            matched_depth_path, gap_ms = find_nearest_depth(
                color_timestamp, depth_index
            )
            if gap_ms <= MAX_TIMESTAMP_GAP_MS:
                cache_key = str(matched_depth_path)
                if cache_key not in depth_cache:
                    depth_cache[cache_key] = cv2.imread(
                        cache_key, cv2.IMREAD_UNCHANGED
                    )
                depth_image = depth_cache[cache_key]
            else:
                rejected_gap_count += 1

        entry["depth_image_id"] = (
            matched_depth_path.name
            if matched_depth_path is not None and depth_image is not None
            else None
        )
        entry["color_depth_gap_ms"] = (
            round(float(gap_ms), 1) if gap_ms is not None else None
        )

        for detection in detections:
            detection["position_3d"] = None

            # localizes accepted real detections
            if not detection.get("is_real", False):
                rejected_decoy_count += 1
                continue

            accepted_real += 1
            if depth_image is not None:
                detection["position_3d"] = localize_detection(
                    detection, depth_image, intrinsics
                )

            if detection["position_3d"] is None:
                no_position_count += 1
            else:
                localized_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2)

    print(f"\n--- Module 3: 3D Localization ({split}) ---")
    print(f"Frames: {len(entries)}")
    print(f"Depth PNGs indexed: {len(depth_index)}")
    print(f"Total detections: {total_detections}")
    print(f"Accepted real detections: {accepted_real}")
    print(f"Accepted real detections localized: {localized_count}")
    print(f"Accepted real detections without position: {no_position_count}")
    print(f"Decoys left without position: {rejected_decoy_count}")
    print(f"Timestamp gaps over {MAX_TIMESTAMP_GAP_MS:.0f} ms: {rejected_gap_count}")
    print(f"Frames with missing timestamps: {missing_timestamp_count}")
    print(f"Central depth patch: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Depth scale: {DEPTH_SCALE} (raw millimetres to metres)")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate 3D camera-frame positions for filtered real detections."
    )
    parser.add_argument(
        "--split",
        choices=("test", "eval"),
        default="test",
        help="Dataset split to process (default: test).",
    )
    args = parser.parse_args()
    run(args.split)


if __name__ == "__main__":
    main()
