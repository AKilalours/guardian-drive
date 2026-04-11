"""
acquisition/visual_odometry.py
Guardian Drive -- Visual Odometry from nuScenes Ego Poses

Frame-to-frame ego motion estimation using real nuScenes poses.
Built by Akila Lourdes Miriyala Francis & Akilan Manivannan
"""
from __future__ import annotations
import json, math, numpy as np
from pathlib import Path
from typing import Optional

class VisualOdometry:
    def __init__(self, data_root="datasets/nuscenes", version="v1.0-mini"):
        self.root = Path(data_root)/version
        self._poses = []
        self._load()

    def _load(self):
        ep = json.loads((self.root/"ego_pose.json").read_text())
        self._poses = sorted(ep, key=lambda e: e["timestamp"])
        print(f"[VO] Loaded {len(self._poses)} ego poses")

    @staticmethod
    def _quat_to_yaw(q):
        w,x,y,z = float(q[0]),float(q[1]),float(q[2]),float(q[3])
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    @staticmethod
    def _quat_to_R(q):
        w,x,y,z = float(q[0]),float(q[1]),float(q[2]),float(q[3])
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    def compute_odometry(self, idx: int) -> Optional[dict]:
        if idx < 1 or idx >= len(self._poses): return None
        p1,p2 = self._poses[idx-1], self._poses[idx]
        t1 = np.array(p1["translation"][:2])
        t2 = np.array(p2["translation"][:2])
        R1 = self._quat_to_R(p1["rotation"])
        t_rel = R1.T @ (np.array(p2["translation"]) - np.array(p1["translation"]))
        dt = (p2["timestamp"]-p1["timestamp"])/1e6
        if dt <= 0: return None
        dx,dy = t2-t1
        dist = math.sqrt(dx**2+dy**2)
        yaw2 = self._quat_to_yaw(p2["rotation"])
        yaw1 = self._quat_to_yaw(p1["rotation"])
        return {
            "frame_idx": idx,
            "dx_m": round(float(dx),4),
            "dy_m": round(float(dy),4),
            "distance_m": round(dist,4),
            "velocity_ms": round(dist/dt,3),
            "velocity_kph": round(dist/dt*3.6,2),
            "heading_deg": round(math.degrees(yaw2),2),
            "heading_change": round(math.degrees(yaw2-yaw1),3),
            "dt_sec": round(dt,4),
            "translation": t_rel.tolist(),
        }

    def reconstruct_trajectory(self, max_frames=200):
        traj = []
        x,y,heading = 0.0,0.0,0.0
        for i in range(1, min(max_frames, len(self._poses))):
            odo = self.compute_odometry(i)
            if not odo: continue
            x += odo["dx_m"]; y += odo["dy_m"]
            heading = odo["heading_deg"]
            traj.append({"frame":i,"x":round(x,3),"y":round(y,3),
                         "heading":round(heading,2),"velocity_kph":odo["velocity_kph"]})
        return traj

if __name__ == "__main__":
    vo = VisualOdometry()
    traj = vo.reconstruct_trajectory(50)
    print(f"Reconstructed {len(traj)} frames")
    for t in traj[:5]:
        print(f"  frame={t['frame']:3d}  x={t['x']:8.2f}m  y={t['y']:8.2f}m  v={t['velocity_kph']:5.1f}kph")
