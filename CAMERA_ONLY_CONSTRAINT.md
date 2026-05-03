# Guardian Drive -- Camera-Only Constraint Analysis

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis

## Tesla Camera-Only Philosophy

Tesla FSD uses camera-only perception -- no lidar, no radar (deprecated).
This is fundamentally different from nuScenes which uses:
- 1x lidar (Velodyne HDL-32E)
- 5x radar
- 6x cameras
- Lidar-assisted 3D annotation labels

## How Guardian Drive Addresses This

### nuScenes BEV -- What We Use vs What Tesla Does

| Aspect | nuScenes (what we use) | Tesla FSD |
|--------|----------------------|-----------|
| Annotation labels | Lidar-assisted 3D boxes | Camera-only auto-labels (Dojo) |
| Depth estimation | Lidar ground truth | Monocular depth prediction |
| Sensor fusion | Camera + Lidar + Radar | Camera only |
| Scale | 1,000 scenes | Billions of miles |
| Format | Public JSON | Proprietary |

### What We Do Honestly

Guardian Drive uses nuScenes mini (10 scenes) for:
1. BEV object streaming -- ego-relative coordinates from known annotations
2. Visual odometry -- frame-to-frame motion from known ego poses
3. Pose-based occupancy mapping -- from known pose records
4. Waypoint prediction -- imitation learning from ego pose sequences

We use nuScenes as the best available public proxy for fleet AV data.
We do NOT claim to replicate Tesla's proprietary data pipeline.

### Camera-Only BEV Lifting (Future Work)

The correct Tesla-aligned approach for BEV depth estimation:
1. Remove all lidar supervision from BEV lifter
2. Force monocular depth prediction from camera features alone
3. Use Lift-Splat-Shoot (LSS) with no depth ground truth
4. Evaluate BEV detection AP with pure camera-predicted depth

This is documented as Gap 2 in our system analysis and is
required future work before making camera-only BEV claims.

## What This Means for Our Results

- nuScenes BEV streaming: valid as AV context proxy
- BEV object counts for risk modulation: valid
- Depth estimates: use lidar-assisted labels (honest limitation)
- Camera-only monocular depth: NOT yet implemented

## Resume Language

Correct:
"Integrated real nuScenes mini AV perception data (18,538 3D annotations)
as environmental context for driver monitoring risk modulation.
Acknowledged lidar-assisted label limitation; documented camera-only
monocular depth estimation as required future work aligned with
Tesla FSD camera-only philosophy."

Incorrect (do not claim):
"Implemented Tesla-style camera-only BEV perception"
