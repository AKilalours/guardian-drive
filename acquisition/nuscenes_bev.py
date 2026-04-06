
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Optional

_CAT_MAP = {
    "human.pedestrian.adult":"pedestrian","human.pedestrian.child":"pedestrian",
    "human.pedestrian.construction_worker":"pedestrian","human.pedestrian.police_officer":"pedestrian",
    "vehicle.car":"car","vehicle.truck":"truck","vehicle.bus.rigid":"bus","vehicle.bus.bendy":"bus",
    "vehicle.motorcycle":"motorcycle","vehicle.bicycle":"bicycle",
    "vehicle.construction":"truck","vehicle.emergency.ambulance":"car","vehicle.emergency.police":"car",
    "movable_object.barrier":"barrier","movable_object.trafficcone":"cone",
}
_COLORS = {
    "car":"#00d4ff","truck":"#ff8c00","bus":"#ff8c00","pedestrian":"#00ff88",
    "motorcycle":"#9b4fff","bicycle":"#9b4fff","barrier":"#ffd600","cone":"#ffd600","unknown":"#4a6a8a",
}

class NuScenesBEV:
    def __init__(self, data_root="datasets/nuscenes", version="v1.0-mini"):
        self.root = Path(data_root)/version
        self._loaded = False
        self._scenes = []; self._scene_idx = 0; self._sample_idx = 0
        self._scene_samples = []; self._last_ts = 0.0; self._frame_dt = 0.5
        self._load()

    def _load(self):
        try:
            samples   = json.loads((self.root/"sample.json").read_text())
            anns      = json.loads((self.root/"sample_annotation.json").read_text())
            ego_poses = json.loads((self.root/"ego_pose.json").read_text())
            instances = json.loads((self.root/"instance.json").read_text())
            cats      = json.loads((self.root/"category.json").read_text())
            scenes    = json.loads((self.root/"scene.json").read_text())
            logs      = json.loads((self.root/"log.json").read_text())

            # Index
            self._samples   = {s["token"]:s for s in samples}
            self._instances = {i["token"]:i for i in instances}
            self._cats      = {c["token"]:c["name"] for c in cats}
            self._logs      = {l["token"]:l for l in logs}

            # ego_pose by timestamp (closest match to sample timestamp)
            self._ego_by_ts = sorted(ego_poses, key=lambda e: e["timestamp"])
            self._ego_ts    = [e["timestamp"] for e in self._ego_by_ts]

            # annotations by sample
            self._anns = {}
            for a in anns:
                self._anns.setdefault(a["sample_token"],[]).append(a)

            # scenes with ordered samples
            self._scenes = []
            for sc in scenes:
                ordered = []
                tok = sc["first_sample_token"]
                while tok:
                    s = self._samples.get(tok)
                    if not s: break
                    ordered.append(tok); tok = s.get("next","")
                log = self._logs.get(sc.get("log_token",""),{})
                self._scenes.append({"name":sc["name"],"location":log.get("location","unknown"),"samples":ordered})

            self._scene_samples = self._scenes[0]["samples"]
            self._loaded = True
            total_anns = sum(len(v) for v in self._anns.values())
            print(f"[nuScenes] {len(self._scenes)} scenes, {len(samples)} samples, {total_anns} annotations OK")
        except Exception as e:
            print(f"[nuScenes] Load failed: {e}"); import traceback; traceback.print_exc()

    @property
    def available(self): return self._loaded

    def _closest_ego(self, timestamp):
        import bisect
        idx = bisect.bisect_left(self._ego_ts, timestamp)
        if idx == 0: return self._ego_by_ts[0]
        if idx >= len(self._ego_by_ts): return self._ego_by_ts[-1]
        before = self._ego_by_ts[idx-1]; after = self._ego_by_ts[idx]
        return before if (timestamp-before["timestamp"]) < (after["timestamp"]-timestamp) else after

    @staticmethod
    def _quat_yaw(q):
        if not q or len(q)<4: return 0.0
        w,x,y,z = float(q[0]),float(q[1]),float(q[2]),float(q[3])
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    def next_frame(self):
        if not self._loaded: return None
        now = time.time()
        if (now-self._last_ts) < self._frame_dt: return None
        self._last_ts = now

        self._sample_idx += 1
        if self._sample_idx >= len(self._scene_samples):
            self._scene_idx = (self._scene_idx+1) % len(self._scenes)
            self._scene_samples = self._scenes[self._scene_idx]["samples"]
            self._sample_idx = 0

        stok = self._scene_samples[self._sample_idx]
        sample = self._samples[stok]
        scene  = self._scenes[self._scene_idx]

        # Get ego pose by timestamp
        ep = self._closest_ego(sample["timestamp"])
        et = ep["translation"]
        ego_x, ego_y = float(et[0]), float(et[1])
        ego_heading = self._quat_yaw(ep["rotation"])

        cos_h = math.cos(-ego_heading)
        sin_h = math.sin(-ego_heading)

        objects = []
        for ann in self._anns.get(stok, []):
            inst = self._instances.get(ann["instance_token"],{})
            cat_name = self._cats.get(inst.get("category_token",""),"")
            cls = _CAT_MAP.get(cat_name)
            if not cls: continue

            tr = ann["translation"]
            dx = float(tr[0])-ego_x; dy = float(tr[1])-ego_y
            rx =  dx*cos_h - dy*sin_h
            ry =  dx*sin_h + dy*cos_h
            dist = math.sqrt(rx**2+ry**2)
            if dist > 50: continue

            obj_heading = self._quat_yaw(ann["rotation"]) - ego_heading
            sz = ann["size"]

            objects.append({
                "x":round(rx,2),"y":round(ry,2),
                "heading":round(obj_heading,3),
                "vx":0.0,"vy":0.0,"speed":0.0,
                "cls":cls,"color":_COLORS.get(cls,"#4a6a8a"),
                "w":round(float(sz[0]),2),"l":round(float(sz[1]),2),
                "dist":round(dist,1),"label":cls.upper(),
            })

        objects.sort(key=lambda o:o["dist"])
        return {
            "ego_x":round(ego_x,2),"ego_y":round(ego_y,2),
            "ego_heading":round(ego_heading,4),
            "objects":objects,"n_objects":len(objects),
            "scene_name":scene["name"],"scene_location":scene["location"],
            "sample_idx":self._sample_idx,"total_samples":len(self._scene_samples),
            "source":"nuscenes_mini",
        }

if __name__=="__main__":
    bev = NuScenesBEV()
    if bev.available:
        time.sleep(0.6)
        for i in range(3):
            time.sleep(0.6)
            f = bev.next_frame()
            if f:
                print(f"Scene:{f['scene_name']} ({f['scene_location']}) sample {f['sample_idx']}/{f['total_samples']}")
                print(f"Ego:({f['ego_x']:.1f},{f['ego_y']:.1f}) objects:{f['n_objects']}")
                for o in f['objects'][:4]:
                    print(f"  {o['cls']:12s} x={o['x']:6.1f}m y={o['y']:6.1f}m dist={o['dist']:.1f}m")
