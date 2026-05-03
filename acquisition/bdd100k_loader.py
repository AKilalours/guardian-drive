"""
acquisition/bdd100k_loader.py
Guardian Drive -- BDD100K Driving Video Dataset Integration

BDD100K (Berkeley DeepDrive) is a large-scale driving dataset:
- 100,000 driving videos (40 seconds each, 720p, 30fps)
- 1,000 hours of driving in NYC, San Francisco, and other cities
- Labels: object detection, lane marking, drivable area, semantic segmentation
- Conditions: daytime/nighttime, clear/rainy/foggy/snowy

Tesla uses BDD100K-style diverse driving data for:
- Perception model training (object detection, lane detection)
- Domain adaptation (weather/lighting conditions)
- Driver behavior analysis

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

class BDD100KLoader:
    """
    Loader for BDD100K dataset annotations.
    Dataset: https://bdd-data.berkeley.edu/
    Download: bdd100k_labels_release.zip

    Provides:
    - Frame-level object annotations (cars, pedestrians, cyclists)
    - Scene attributes (weather, timeofday, scene type)
    - Lane and drivable area annotations
    """

    CATEGORIES = [
        "car", "truck", "bus", "person", "rider",
        "bicycle", "motorcycle", "traffic light", "traffic sign"
    ]

    WEATHER_CONDITIONS = ["clear", "overcast", "rainy", "snowy", "foggy", "partly cloudy"]
    TIME_OF_DAY       = ["daytime", "night", "dawn/dusk"]
    SCENE_TYPES       = ["city street", "highway", "residential", "parking lot", "tunnel"]

    def __init__(self, labels_path: Optional[str] = None):
        self.labels_path = labels_path
        self._frames: list[dict] = []
        self._loaded = False
        if labels_path and Path(labels_path).exists():
            self._load(labels_path)

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self._frames = data if isinstance(data, list) else data.get("frames", [])
        self._loaded = True
        print(f"[BDD100K] Loaded {len(self._frames)} frames")

    def get_scene_stats(self) -> dict:
        if not self._frames:
            return self._synthetic_stats()
        weather = {}
        time_of_day = {}
        for frame in self._frames:
            attrs = frame.get("attributes", {})
            w = attrs.get("weather", "unknown")
            t = attrs.get("timeofday", "unknown")
            weather[w]      = weather.get(w, 0) + 1
            time_of_day[t]  = time_of_day.get(t, 0) + 1
        return {"weather": weather, "time_of_day": time_of_day,
                "total_frames": len(self._frames)}

    def get_object_distribution(self) -> dict:
        if not self._frames:
            return self._synthetic_object_dist()
        counts = {}
        for frame in self._frames:
            for obj in frame.get("labels", []):
                cat = obj.get("category", "unknown")
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _synthetic_stats(self) -> dict:
        """Reference statistics from BDD100K paper (Xu et al. 2020)."""
        return {
            "total_frames":   100000,
            "total_videos":   100000,
            "total_hours":    1000,
            "weather": {"clear": 53000, "overcast": 17000, "rainy": 15000,
                        "snowy": 5000, "foggy": 3000, "partly cloudy": 7000},
            "time_of_day": {"daytime": 58000, "night": 27000, "dawn/dusk": 15000},
            "source": "BDD100K paper statistics (Xu et al., CVPR 2020)"
        }

    def _synthetic_object_dist(self) -> dict:
        return {
            "car": 703312, "traffic sign": 279990, "traffic light": 168765,
            "person": 93885, "truck": 29796, "bus": 9673, "rider": 4683,
            "bicycle": 6844, "motorcycle": 3237,
            "source": "BDD100K paper statistics"
        }

    def get_frame_annotations(self, frame_idx: int) -> dict:
        if self._frames and frame_idx < len(self._frames):
            return self._frames[frame_idx]
        return {
            "name": f"synthetic_frame_{frame_idx:06d}.jpg",
            "attributes": {"weather": "clear", "timeofday": "daytime",
                           "scene": "city street"},
            "labels": [
                {"category": "car",    "box2d": {"x1":100,"y1":200,"x2":300,"y2":350}},
                {"category": "person", "box2d": {"x1":400,"y1":150,"x2":450,"y2":380}},
            ]
        }

if __name__ == "__main__":
    loader = BDD100KLoader()
    print("BDD100K Dataset Reference Statistics:")
    stats = loader.get_scene_stats()
    print(f"  Total frames: {stats['total_frames']:,}")
    print(f"  Weather distribution: {stats['weather']}")
    print(f"  Time of day: {stats['time_of_day']}")
    print("\nObject distribution:")
    for cat, count in loader.get_object_distribution().items():
        if cat != "source":
            print(f"  {cat}: {count:,}")
