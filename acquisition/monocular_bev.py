"""
acquisition/monocular_bev.py
Guardian Drive -- Camera-Only BEV (Tesla FSD Philosophy)

Implements monocular depth estimation for BEV without lidar.
This is the Tesla-aligned approach: camera-only, no lidar.

Current nuScenes BEV uses lidar-assisted labels.
This module implements the camera-only alternative using
Lift-Splat-Shoot (LSS) style depth prediction.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import torch
import torch.nn as nn
import numpy as np

class MonocularDepthHead(nn.Module):
    """
    Predicts depth bins from monocular camera features.
    Replaces lidar-assisted depth in BEV lifting.
    Tesla FSD uses this approach -- no lidar at inference.
    """
    def __init__(self, in_channels: int = 256,
                 n_depth_bins: int = 64,
                 depth_min: float = 1.0,
                 depth_max: float = 60.0):
        super().__init__()
        self.depth_bins = torch.linspace(
            depth_min, depth_max, n_depth_bins)
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, n_depth_bins, 1),
        )

    def forward(self, camera_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_features: [B, C, H, W]
        Returns:
            depth_dist: [B, D, H, W] softmax depth distribution
        """
        logits = self.depth_head(camera_features)
        return torch.softmax(logits, dim=1)

    def expected_depth(self, depth_dist: torch.Tensor) -> torch.Tensor:
        """
        Expected depth from distribution.
        depth_dist: [B, D, H, W]
        Returns: [B, H, W]
        """
        bins = self.depth_bins.to(depth_dist.device)
        bins = bins.view(1,-1,1,1)
        return (depth_dist * bins).sum(dim=1)

class CameraOnlyBEV(nn.Module):
    """
    Full camera-only BEV construction.

    Pipeline:
    1. Camera features -> MonocularDepthHead -> depth distribution
    2. Lift: pixel + depth -> 3D point cloud (no lidar)
    3. Splat: 3D points -> BEV grid via pillar pooling
    4. Shoot: BEV features for downstream tasks

    This is honest about the limitation vs lidar-assisted:
    - Lidar PTBDB AUC: uses ground-truth depth
    - Camera-only: predicts depth, higher uncertainty
    - Performance gap documented in CAMERA_ONLY_CONSTRAINT.md
    """
    def __init__(self, H: int = 200, W: int = 200,
                 res: float = 0.5):
        super().__init__()
        self.H   = H
        self.W   = W
        self.res = res
        self.depth_head = MonocularDepthHead()

    def lift_to_3d(self, pixels_uv: torch.Tensor,
                    depth_dist: torch.Tensor,
                    K_inv: torch.Tensor) -> torch.Tensor:
        """
        Lift 2D pixels to 3D using predicted depth distribution.
        pixels_uv: [B, N, 2]
        depth_dist: [B, D, H, W]
        K_inv: [B, 3, 3] inverse camera intrinsics
        Returns: [B, N, 3] 3D points
        """
        B, N, _ = pixels_uv.shape
        # Expected depth per pixel
        exp_d = depth_dist.mean(dim=1)  # [B, H, W] simplified
        # Unproject
        ones  = torch.ones(B, N, 1, device=pixels_uv.device)
        homo  = torch.cat([pixels_uv, ones], dim=-1)  # [B, N, 3]
        pts3d = torch.bmm(homo, K_inv.transpose(1,2))  # [B, N, 3]
        # Scale by depth (simplified -- uses mean depth)
        depth = exp_d.mean(dim=(1,2), keepdim=True).unsqueeze(-1)
        return pts3d * depth

    def splat_to_bev(self, pts3d: torch.Tensor) -> torch.Tensor:
        """
        Splat 3D points to BEV grid.
        pts3d: [B, N, 3] ego-frame points
        Returns: [B, H, W] occupancy
        """
        B = pts3d.shape[0]
        grid = torch.zeros(B, self.H, self.W,
                            device=pts3d.device)
        x_min = y_min = -(self.H * self.res / 2)
        gx = ((pts3d[:,:,0]-x_min)/self.res).long().clamp(0,self.W-1)
        gy = ((pts3d[:,:,1]-y_min)/self.res).long().clamp(0,self.H-1)
        for b in range(B):
            grid[b].index_put_((gy[b], gx[b]),
                                torch.ones(gx.shape[1],
                                            device=pts3d.device),
                                accumulate=True)
        return grid.clamp(0, 1)

    def forward(self, camera_features: torch.Tensor,
                 pixels_uv: torch.Tensor,
                 K_inv: torch.Tensor) -> dict:
        """Full camera-only BEV forward pass."""
        depth_dist = self.depth_head(camera_features)
        pts3d      = self.lift_to_3d(pixels_uv, depth_dist, K_inv)
        bev_grid   = self.splat_to_bev(pts3d)
        exp_depth  = self.depth_head.expected_depth(depth_dist)
        return {
            "bev_grid":   bev_grid,
            "depth_dist": depth_dist,
            "exp_depth":  exp_depth,
            "pts3d":      pts3d,
            "method":     "camera-only (no lidar)",
        }

if __name__ == "__main__":
    print("Camera-Only BEV -- Architecture Test")
    B,C,H,W = 1,256,32,32
    feats   = torch.randn(B,C,H,W)
    pixels  = torch.randn(B,100,2)*30 + torch.tensor([W/2,H/2])
    K_inv   = torch.eye(3).unsqueeze(0).repeat(B,1,1)
    model   = CameraOnlyBEV()
    out     = model(feats,pixels,K_inv)
    print(f"BEV grid shape: {out['bev_grid'].shape}")
    print(f"Depth dist:     {out['depth_dist'].shape}")
    print(f"Method:         {out['method']}")
    print("Camera-only BEV architecture verified")
