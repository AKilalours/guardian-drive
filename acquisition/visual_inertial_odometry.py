"""
acquisition/visual_inertial_odometry.py
Guardian Drive -- Visual-Inertial Odometry (VIO)

Extends real monocular SLAM (slam_real.json: 1,316 map points, 99.7% tracking)
with IMU preintegration to resolve scale ambiguity.

Monocular SLAM limitation: unit-scale trajectory (no metric scale).
IMU fix: preintegrate accelerometer + gyroscope to get metric displacement,
         then correct visual scale estimate.

Status:
  - Visual SLAM:     VERIFIED on MacBook webcam (slam_real.json)
  - IMU component:   IMPLEMENTED -- needs real IMU hardware for validation
  - Scale correction: IMPLEMENTED -- needs real IMU data for validation

Reference: Forster et al., "IMU Preintegration on Manifold for
Efficient Visual-Inertial Maximum-a-Posteriori Estimation", RSS 2015.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── IMU Data Types ────────────────────────────────────────────────────────────

@dataclass
class IMUMeasurement:
    """Single IMU sample."""
    timestamp: float
    accel:     np.ndarray   # [ax, ay, az] m/s^2 (corrected for gravity)
    gyro:      np.ndarray   # [wx, wy, wz] rad/s


@dataclass
class IMUPreintegration:
    """
    Preintegrated IMU measurement between two keyframes.
    Integrates without re-linearization error.
    """
    dt:       float
    delta_R:  np.ndarray   # [3,3] SO3 rotation change
    delta_v:  np.ndarray   # [3]   velocity change  m/s
    delta_p:  np.ndarray   # [3]   position change  m
    cov:      np.ndarray   # [9,9] covariance matrix


# ── IMU Preintegration ────────────────────────────────────────────────────────

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues rotation formula: omega (rad) -> R [3,3]."""
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        return np.eye(3)
    axis = omega / angle
    K    = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)


def preintegrate_imu(measurements: List[IMUMeasurement],
                      dt: float) -> IMUPreintegration:
    """
    Preintegrate IMU samples between two visual keyframes.

    Args:
        measurements: list of IMU samples
        dt:           IMU sample period (seconds)

    Returns:
        IMUPreintegration with delta_R, delta_v, delta_p
    """
    delta_R = np.eye(3)
    delta_v = np.zeros(3)
    delta_p = np.zeros(3)

    for imu in measurements:
        # Position and velocity (using current rotation)
        a_body  = delta_R @ imu.accel
        delta_p = delta_p + delta_v*dt + 0.5*a_body*dt**2
        delta_v = delta_v + a_body*dt
        # Rotation (SO3 exponential map)
        delta_R = delta_R @ so3_exp(imu.gyro * dt)

    # Simple diagonal covariance (would use noise model in production)
    accel_noise = 0.01   # m/s^2/sqrt(Hz) -- typical MEMS
    gyro_noise  = 0.001  # rad/s/sqrt(Hz)
    T           = len(measurements) * dt
    cov         = np.diag([
        accel_noise**2 * T, accel_noise**2 * T, accel_noise**2 * T,
        accel_noise**2 * T, accel_noise**2 * T, accel_noise**2 * T,
        gyro_noise**2  * T, gyro_noise**2  * T, gyro_noise**2  * T,
    ])

    return IMUPreintegration(
        dt      = len(measurements) * dt,
        delta_R = delta_R,
        delta_v = delta_v,
        delta_p = delta_p,
        cov     = cov,
    )


# ── Scale Correction ──────────────────────────────────────────────────────────

def correct_scale(visual_t:   np.ndarray,
                   imu_delta_p: np.ndarray,
                   prev_scale:  float = 1.0,
                   alpha:       float = 0.1) -> float:
    """
    Estimate metric scale from visual/IMU correspondence.
    Resolves monocular SLAM scale ambiguity.

    Args:
        visual_t:    unit-scale visual translation vector
        imu_delta_p: metric IMU displacement (meters)
        prev_scale:  previous scale estimate (EMA state)
        alpha:       EMA smoothing (0=no update, 1=instant update)

    Returns:
        Updated metric scale estimate
    """
    vis_norm = np.linalg.norm(visual_t)
    imu_norm = np.linalg.norm(imu_delta_p)

    if vis_norm < 1e-6:
        return prev_scale    # no visual motion -- keep previous

    new_scale = imu_norm / vis_norm
    # Clamp to reasonable range (0.01m to 100m per unit)
    new_scale = float(np.clip(new_scale, 0.01, 100.0))
    # Exponential moving average for robustness
    return alpha * new_scale + (1 - alpha) * prev_scale


# ── VIO System ────────────────────────────────────────────────────────────────

class VIOSystem:
    """
    Visual-Inertial Odometry System.

    Fuses monocular ORB-SLAM (verified) with IMU preintegration
    to obtain metric-scale trajectory.

    Pipeline per frame:
    1. Visual: ORB features -> Essential matrix -> R, t (unit scale)
    2. IMU:    Preintegrate samples since last keyframe -> delta_p (metric)
    3. Scale:  Correct visual scale using IMU metric displacement (EMA)
    4. Pose:   Accumulate metric trajectory

    Verified component (slam_real.json):
      - 893 frames processed at 30fps
      - 890/893 tracked (99.7%)
      - 1,316 3D map points
      - 198.3 mean ORB inliers/frame

    Pending (needs real IMU hardware):
      - Metric scale validation
      - IMU bias estimation
      - Full EKF/factor graph fusion
    """

    def __init__(self, K: np.ndarray, imu_dt: float = 0.005):
        """
        Args:
            K:      [3,3] camera intrinsics
            imu_dt: IMU sample period (200Hz = 0.005s typical)
        """
        self.K          = K
        self.imu_dt     = imu_dt
        self.scale      = 1.0        # metric scale estimate
        self.R_total    = np.eye(3)  # accumulated rotation
        self.t_total    = np.zeros(3)# accumulated position (metric)
        self.trajectory = [np.zeros(3)]
        self.n_frames   = 0
        self.n_imu_fused= 0

    def process_frame(
        self,
        R_visual:         np.ndarray,
        t_visual:         np.ndarray,
        imu_measurements: Optional[List[IMUMeasurement]] = None,
    ) -> dict:
        """
        Process one visual frame with optional IMU fusion.

        Args:
            R_visual:         [3,3] rotation from Essential matrix
            t_visual:         [3]   unit-scale translation
            imu_measurements: IMU samples since last frame (or None)

        Returns:
            dict with position, scale, source, tracking info
        """
        source = "visual_only"

        if imu_measurements and len(imu_measurements) > 0:
            preint = preintegrate_imu(imu_measurements, self.imu_dt)
            self.scale = correct_scale(
                t_visual, preint.delta_p, self.scale)
            source = "VIO"
            self.n_imu_fused += 1

        # Apply metric scale and accumulate trajectory
        t_metric     = t_visual * self.scale
        self.R_total = R_visual @ self.R_total
        self.t_total = self.t_total + self.R_total.T @ t_metric
        self.trajectory.append(self.t_total.copy())
        self.n_frames += 1

        return {
            "frame":    self.n_frames,
            "position": [round(x, 4) for x in self.t_total.tolist()],
            "scale":    round(self.scale, 4),
            "source":   source,
        }

    def summary(self) -> dict:
        """Return trajectory summary."""
        traj  = np.array(self.trajectory)
        diffs = np.diff(traj, axis=0)
        length = float(np.linalg.norm(diffs, axis=1).sum())
        return {
            "total_frames":    self.n_frames,
            "imu_fused":       self.n_imu_fused,
            "visual_only":     self.n_frames - self.n_imu_fused,
            "final_scale":     round(self.scale, 4),
            "trajectory_pts":  len(self.trajectory),
            "path_length_m":   round(length, 3),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Guardian Drive -- VIO Architecture Demo")
    print("=" * 55)
    print("Visual SLAM:    VERIFIED (slam_real.json: 1,316 pts)")
    print("IMU component:  IMPLEMENTED -- synthetic data only")
    print("Scale correct:  EMA from IMU/visual correspondence\n")

    # Apple M4 MacBook camera approximate intrinsics
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,  0,   1]], dtype=np.float64)

    vio = VIOSystem(K, imu_dt=0.005)   # 200Hz IMU
    np.random.seed(42)

    print("Simulating 20 frames (synthetic IMU, forward motion):")
    for i in range(20):
        # Synthetic visual pose (small forward motion)
        R = so3_exp(np.array([0, np.radians(1), 0]))   # 1deg yaw
        t = np.array([0.1, 0.0, 0.5])                  # unit scale

        # Synthetic IMU at 200Hz (20 samples between frames at 10fps)
        imu = [IMUMeasurement(
            timestamp = i*0.1 + j*0.005,
            accel     = np.array([0.0, 0.0, 0.0]),   # level motion
            gyro      = np.array([0.0, np.radians(10), 0.0])  # 10 deg/s yaw
        ) for j in range(20)]

        result = vio.process_frame(R, t, imu)
        if i % 5 == 0:
            print(f"  frame={result['frame']:3d} "
                  f"pos={result['position']} "
                  f"scale={result['scale']} "
                  f"src={result['source']}")

    summary = vio.summary()
    print(f"\nVIO Summary:")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    print("\nNext steps for real VIO:")
    print("  1. Connect IMU (MPU-9250 or similar at 200Hz)")
    print("  2. Time-synchronize camera (30fps) and IMU (200Hz)")
    print("  3. Estimate IMU bias via stationary calibration")
    print("  4. Replace EMA scale with EKF or factor graph")
    print("  5. Compare trajectory vs ground truth (KITTI/EuRoC)")
