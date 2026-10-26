"""Shared depth helpers (Module 2 decoy filter and Module 3 3D localization).

Depth frames are already registered to the colour camera: the depth camera_info
has frame_id camera_color_optical_frame and the same K as the colour stream,
so a colour-pixel box can be indexed straight into the depth image.
Depth PNGs are uint16 in millimetres; 0 means no measurement.
"""
import re
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

TS_RE = re.compile(r"_(\d{19})\.(png|txt)$")


def timestamp(path_or_id):
    """Nanosecond timestamp from a file name or image id (camera_color_image_<ns>)."""
    m = re.search(r"(\d{19})", str(path_or_id))
    if not m:
        raise ValueError(f"no timestamp in {path_or_id}")
    return int(m.group(1))


def load_intrinsics(camera_info_txt):
    """Return K (3x3), width, height from a ROS camera_info text dump."""
    txt = Path(camera_info_txt).read_text()
    k = [float(v) for v in re.search(r"^K: \[(.*?)\]", txt, re.M).group(1).split(",")]
    w = int(re.search(r"^width: (\d+)", txt, re.M).group(1))
    h = int(re.search(r"^height: (\d+)", txt, re.M).group(1))
    return np.array(k).reshape(3, 3), w, h


def split_intrinsics(raw_dir):
    """K of the colour camera for one recording (constant over the recording)."""
    first = sorted(Path(raw_dir, "camera_color_camera_info").glob("*.txt"))[0]
    return load_intrinsics(first)


class DepthIndex:
    """Nearest-timestamp lookup of depth frames for one recording."""

    def __init__(self, raw_dir):
        paths = sorted(Path(raw_dir, "camera_depth_image_raw").glob("camera_depth_image_*.png"))
        self.ts = [timestamp(p.name) for p in paths]
        self.paths = paths

    def match(self, color_ts, max_gap_ms=100.0):
        """(path, gap_ms) of the nearest depth frame; path is None if gap > max_gap_ms."""
        i = bisect_left(self.ts, color_ts)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.ts)]
        j = min(cands, key=lambda j: abs(self.ts[j] - color_ts))
        gap = abs(self.ts[j] - color_ts) / 1e6
        return (self.paths[j] if gap <= max_gap_ms else None), gap


def load_depth_m(path):
    """Depth image in metres, float32, NaN where there is no measurement."""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None or d.dtype != np.uint16:
        raise ValueError(f"unexpected depth image {path}")
    z = d.astype(np.float32) / 1000.0
    z[d == 0] = np.nan
    return z


def backproject(u, v, z, K):
    """Pixel coordinates + depth (m) -> camera-frame points (x right, y down, z forward), metres."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
