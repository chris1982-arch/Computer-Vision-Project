"""Module 3 - figure: one eval frame with its accepted detection and 3D position.

Run from project root: python src/plot_3d_example.py
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_IMAGE_ID = "camera_color_image_1727164260934242393"


def load_frames(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("frames", "results", "images", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if "image_id" in payload:
            return [payload]
    raise ValueError("Could not find a list of frames in the localization JSON.")


def find_image(data_root, image_id, frame=None):
    if frame:
        image_path = frame.get("image_path")
        if image_path:
            p = Path(image_path)
            if p.exists():
                return p
            matches = list(data_root.rglob(p.name))
            if matches:
                return matches[0]

    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        direct = data_root / f"{image_id}{ext}"
        if direct.exists():
            return direct

    matches = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        matches.extend(data_root.rglob(f"{image_id}{ext}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {image_id} under {data_root}. "
        "Pass --data-root pointing to the folder containing the eval images."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="outputs/localized_eval.json")
    parser.add_argument("--data-root", default="data/raw/eval")
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--out", default="outputs/figures/3d_localization_example.png")
    args = parser.parse_args()

    json_path = Path(args.json)
    data_root = Path(args.data_root)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    frames = load_frames(json_path)
    frame = next((f for f in frames if f.get("image_id") == args.image_id), None)
    if frame is None:
        raise KeyError(f"Image ID {args.image_id!r} not found in {json_path}")

    image_path = find_image(data_root, args.image_id, frame)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    detections = frame.get("detections", [])
    accepted = [
        d for d in detections
        if d.get("is_real") is True and isinstance(d.get("position_3d"), dict)
    ]
    if not accepted:
        raise ValueError("This frame has no accepted real detection with position_3d.")


    chosen = max(accepted, key=lambda d: float(d.get("confidence", 0.0)))
    pos = chosen["position_3d"]
    x, y, z = float(pos["x"]), float(pos["y"]), float(pos["z"])
    bbox = chosen.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        raise ValueError("Selected detection has no valid bbox_xyxy.")
    x1, y1, x2, y2 = map(float, bbox)
    target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2

    fig, (ax_img, ax_info) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax_img.imshow(image)
    ax_img.set_title("Detection and decoy rejection")
    ax_img.axis("off")


    legend_lines = []
    for idx, det in enumerate(detections, start=1):
        box = det.get("bbox_xyxy")
        if not box or len(box) != 4:
            continue
        bx1, by1, bx2, by2 = map(float, box)
        is_accepted = det.get("is_real") is True
        color = "limegreen" if is_accepted else "red"
        is_chosen = det is chosen
        ax_img.add_patch(Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            fill=False, edgecolor=color,
            linewidth=3.0 if is_chosen else 1.8
        ))

        marker_y = max(15, by1 - 4)
        ax_img.text(
            bx1, marker_y, str(idx),
            color="white", fontsize=9, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="circle,pad=0.22", facecolor=color,
                      edgecolor="white", linewidth=0.6)
        )
        conf = det.get("confidence")
        status = "ACCEPTED REAL" if is_accepted else "REJECTED"
        legend_lines.append(
            (idx, color, status, float(conf) if conf is not None else None, is_chosen)
        )



    callout_x = max(12, min(width * 0.04, width - 280))
    callout_y = max(18, min(height * 0.08, height - 100))
    callout_text = f"Estimated 3D position (m)\nX: {x:.3f}   Y: {y:.3f}   Z: {z:.3f}"
    ax_img.annotate(
        callout_text,
        xy=(target_x, target_y),
        xycoords="data",
        xytext=(callout_x, callout_y),
        textcoords="data",
        color="black",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#eaf7ea",
                  edgecolor="green", alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="green", linewidth=1.8,
                        connectionstyle="arc3,rad=0.12"),
        zorder=10,
    )


    ax_info.axis("off")
    ax_info.set_title("3D localization", pad=14, fontweight="bold")
    ax_info.text(
        0.03, 0.92,
        f"X (right):   {x:.3f} m\nY (down):   {y:.3f} m\nZ (forward): {z:.3f} m",
        transform=ax_info.transAxes, fontsize=13, va="top", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="#eaf7ea",
                  edgecolor="green", alpha=1.0)
    )
    ax_info.text(
        0.03, 0.68,
        "Coordinate frame:\ncamera_color_optical_frame",
        transform=ax_info.transAxes, fontsize=9, va="top"
    )
    ax_info.text(
        0.03, 0.56, "Detection legend",
        transform=ax_info.transAxes, fontsize=11, fontweight="bold", va="top"
    )


    y_pos = 0.50
    for idx, color, status, conf, is_chosen in legend_lines:
        conf_text = f"  conf={conf:.2f}" if conf is not None else ""
        selected_text = "  ← coordinate source" if is_chosen else ""
        ax_info.text(
            0.04, y_pos,
            f"{idx}. {status}{conf_text}{selected_text}",
            transform=ax_info.transAxes, fontsize=8.5, va="top",
            color=color, fontweight="bold" if is_chosen else "normal"
        )
        y_pos -= 0.055
        if y_pos < 0.04:
            ax_info.text(
                0.04, y_pos,
                "… see JSON for remaining detections",
                transform=ax_info.transAxes, fontsize=8, va="top", color="gray"
            )
            break

    fig.suptitle("3D Localization of a Real Fire Extinguisher",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {out_path}")
    print(f"Image: {image_path}")
    print(f"Detection ID: {chosen.get('det_id', '(not provided)')}")
    print(f"Estimated position: X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m")


if __name__ == "__main__":
    main()
