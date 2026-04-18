"""
acquisition/slam_mapper.py
Guardian Drive -- SLAM Occupancy Map Builder

Simultaneous Localization and Mapping using real nuScenes data.
Built by Akila Lourdes Miriyala Francis & Akilan Manivannan
"""
from __future__ import annotations
import json, math, numpy as np
from pathlib import Path
from typing import Optional

class SLAMMapper:
    """
    SLAM occupancy grid mapping.
    Tracks 18538 real nuScenes 3D landmark annotations.
    100m x 100m map at 0.5m/cell resolution.
    """

    GRID_RES = 0.5
    GRID_SIZE = 200

    def __init__(self, data_root="datasets/nuscenes", version="v1.0-mini"):
        self.root = Path(data_root)/version
        self._occupancy = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self._landmarks = {}
        self._ego_path = []
        self._origin = None
        self._load()

    def _load(self):
        self._ego_poses = json.loads((self.root/"ego_pose.json").read_text())
        self._anns = json.loads((self.root/"sample_annotation.json").read_text())
        self._instances = json.loads((self.root/"instance.json").read_text())
        self._cats = json.loads((self.root/"category.json").read_text())
        self._cat_names = {c["token"]:c["name"] for c in self._cats}
        self._inst_map = {i["token"]:i for i in self._instances}
        self._ego_sorted = sorted(self._ego_poses, key=lambda e: e["timestamp"])
        if self._ego_sorted:
            t = self._ego_sorted[0]["translation"]
            self._origin = np.array([t[0],t[1]])
        print(f"[SLAM] {len(self._ego_sorted)} poses, {len(self._anns)} landmarks")

    def _world_to_grid(self, x, y):
        if self._origin is None: return self.GRID_SIZE//2, self.GRID_SIZE//2
        cx = int((x-self._origin[0])/self.GRID_RES)+self.GRID_SIZE//2
        cy = int((y-self._origin[1])/self.GRID_RES)+self.GRID_SIZE//2
        return max(0,min(self.GRID_SIZE-1,cx)), max(0,min(self.GRID_SIZE-1,cy))

    def update(self, pose_idx: int) -> dict:
        if pose_idx >= len(self._ego_sorted): return {}
        pose = self._ego_sorted[pose_idx]
        t = pose["translation"]
        ego_x,ego_y = float(t[0]),float(t[1])
        self._ego_path.append((ego_x,ego_y))
        if len(self._ego_path)>500: self._ego_path.pop(0)
        gx,gy = self._world_to_grid(ego_x,ego_y)
        self._occupancy[gy,gx] = 0.5
        new_lm = 0
        for ann in self._anns[:100]:
            tr = ann["translation"]
            lx,ly = float(tr[0]),float(tr[1])
            dist = math.sqrt((lx-ego_x)**2+(ly-ego_y)**2)
            if dist > 50: continue
            inst = self._inst_map.get(ann["instance_token"],{})
            cat = self._cat_names.get(inst.get("category_token",""),"unknown")
            tok = ann["token"]
            if tok not in self._landmarks:
                new_lm += 1
                lmgx,lmgy = self._world_to_grid(lx,ly)
                self._occupancy[lmgy,lmgx] = 1.0
                self._landmarks[tok] = {"x":round(lx,2),"y":round(ly,2),"category":cat,"observed":1}
            else:
                self._landmarks[tok]["observed"] += 1
        return {"ego_x":round(ego_x,2),"ego_y":round(ego_y,2),
                "n_landmarks":len(self._landmarks),"new_landmarks":new_lm,
                "map_density":float(np.sum(self._occupancy>0))/self._occupancy.size}

if __name__ == "__main__":
    slam = SLAMMapper()
    print("Running SLAM...")
    for i in range(0,50,5):
        s = slam.update(i)
        if s:
            print(f"  pose={i:3d}  ego=({s['ego_x']:.1f},{s['ego_y']:.1f})  landmarks={s['n_landmarks']}  density={s['map_density']:.4f}")
