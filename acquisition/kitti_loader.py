"""
acquisition/kitti_loader.py
Guardian Drive -- KITTI Dataset Integration

KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute):
- 389 stereo sequences, 39,180 frames
- LiDAR point clouds (Velodyne HDL-64E)
- Calibrated stereo cameras + GPS/IMU
- 3D object detection: 80,256 annotated objects
- Depth estimation, optical flow, scene flow, road segmentation

Tesla uses KITTI-style data for:
- Monocular depth estimation training
- 3D object detection evaluation
- Stereo vision calibration
- LiDAR-camera fusion

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class KITTILoader:
    """
    Loader for KITTI autonomous driving dataset.
    Dataset: http://www.cvlibs.net/datasets/kitti/
    """

    OBJECT_CLASSES = [
        "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
        "Cyclist", "Tram", "Misc", "DontCare"
    ]

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root) if data_root else None
        self._loaded = data_root and Path(data_root).exists()

    def parse_calibration(self, calib_path: str) -> dict:
        """Parse KITTI calibration file into projection matrices."""
        calib = {}
        with open(calib_path) as f:
            for line in f:
                if ":" in line:
                    key, vals = line.strip().split(":", 1)
                    calib[key.strip()] = np.array(
                        [float(x) for x in vals.strip().split()])
        # P2: left camera 3x4 projection matrix
        if "P2" in calib:
            calib["P2"] = calib["P2"].reshape(3, 4)
        # Tr_velo_to_cam: LiDAR to camera transform
        if "Tr_velo_to_cam" in calib:
            T = calib["Tr_velo_to_cam"].reshape(3, 4)
            calib["Tr_velo_to_cam"] = np.vstack([T, [0,0,0,1]])
        return calib

    def parse_labels(self, label_path: str) -> list[dict]:
        """Parse KITTI label file into structured annotations."""
        objects = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                objects.append({
                    "class":     parts[0],
                    "truncated": float(parts[1]),
                    "occluded":  int(parts[2]),
                    "alpha":     float(parts[3]),
                    "bbox_2d":   [float(x) for x in parts[4:8]],
                    "dimensions":[float(x) for x in parts[8:11]],  # h,w,l
                    "location":  [float(x) for x in parts[11:14]], # x,y,z camera
                    "rotation_y": float(parts[14]),
                })
        return objects

    def project_lidar_to_image(self, points: np.ndarray,
                                calib: dict) -> np.ndarray:
        """
        Project LiDAR points to image plane using calibration matrices.
        Core operation in LiDAR-camera fusion.
        """
        T_velo = calib.get("Tr_velo_to_cam",
                            np.eye(4))
        P2     = calib.get("P2",
                            np.eye(3,4))

        # Homogeneous LiDAR points [N, 4]
        pts_h = np.hstack([points[:,:3],
                            np.ones((len(points),1))])
        # Transform to camera frame
        pts_cam = (T_velo @ pts_h.T).T  # [N, 4]
        # Keep points in front of camera
        mask    = pts_cam[:, 2] > 0
        pts_cam = pts_cam[mask]
        # Project to image
        pts_img = (P2 @ pts_cam.T).T     # [N, 3]
        pts_img[:, :2] /= pts_img[:, 2:3]
        return pts_img[:, :2], mask

    def compute_depth_map(self, points: np.ndarray,
                           calib: dict,
                           img_shape: Tuple[int,int] = (375, 1242)) -> np.ndarray:
        """
        Create sparse depth map from LiDAR projection.
        Used for monocular depth estimation supervision.
        """
        H, W       = img_shape
        depth_map  = np.zeros((H, W), dtype=np.float32)
        pts_2d, mask = self.project_lidar_to_image(points, calib)
        depths       = points[mask, 2]
        xs = np.clip(pts_2d[:, 0].astype(int), 0, W-1)
        ys = np.clip(pts_2d[:, 1].astype(int), 0, H-1)
        depth_map[ys, xs] = depths
        return depth_map

    def get_dataset_stats(self) -> dict:
        """KITTI dataset reference statistics."""
        return {
            "sequences":         389,
            "frames":            39180,
            "annotated_objects": 80256,
            "object_dist": {
                "Car": 28742, "Van": 2914, "Truck": 1094,
                "Pedestrian": 4487, "Cyclist": 1627, "Tram": 511,
            },
            "sensors": ["Velodyne HDL-64E LiDAR",
                        "2x PointGray Flea2 cameras (stereo)",
                        "GPS/IMU Applanix"],
            "city": "Karlsruhe, Germany",
            "source": "Geiger et al., CVPR 2012"
        }

if __name__ == "__main__":
    loader = KITTILoader()
    stats  = loader.get_dataset_stats()
    print("KITTI Dataset Reference Statistics:")
    print(f"  Sequences:  {stats['sequences']}")
    print(f"  Frames:     {stats['frames']:,}")
    print(f"  Objects:    {stats['annotated_objects']:,}")
    print(f"  Object distribution: {stats['object_dist']}")
    print(f"  Sensors: {stats['sensors']}")
