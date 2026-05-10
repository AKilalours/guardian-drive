"""
scripts/train_all.py
Guardian Drive v2 — Master Training Script

Runs all three upgrade projects:
  1. CARLA closed-loop agent (BC → PPO)
  2. Fleet telemetry pipeline (nuPlan + Waymo ingestion + rare event mining)
  3. BEV perception (BEVFormer integration + nuScenes eval)

Usage:
    python scripts/train_all.py --project all
    python scripts/train_all.py --project carla
    python scripts/train_all.py --project fleet
    python scripts/train_all.py --project bev

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
LIU Brooklyn — MS Artificial Intelligence
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# ─────────────────────────────────────────────
# Project 1: CARLA closed-loop agent
# ─────────────────────────────────────────────

def run_carla_project(save_dir: str = "outputs/carla_agent", device: str = "cpu"):
    print("\n" + "=" * 60)
    print("PROJECT 1: CARLA Closed-Loop Safety Agent")
    print("BC → PPO | nuScenes-style evaluation")
    print("=" * 60)

    from carla_agent.env.carla_env import GuardianDriveCARLAEnv
    from carla_agent.agent.policy import (
        GuardianPolicyNet, BehaviorCloningTrainer, PPOTrainer
    )
    from carla_agent.eval.evaluate import EvaluationRunner

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Environment (runs in simulation mode without CARLA server)
    env = GuardianDriveCARLAEnv(
        fault_injection=True,
        seed=42,
        max_steps=500,
    )

    policy = GuardianPolicyNet()

    # Stage 1: Behavior Cloning
    print("\n[Stage 1] Behavior Cloning from expert demonstrations...")
    bc_trainer = BehaviorCloningTrainer(policy, device=device)
    obs, actions = bc_trainer.collect_demos(env, n_steps=2000)
    print(f"  Collected {len(obs)} expert demonstrations")
    bc_metrics = bc_trainer.train(obs, actions, n_epochs=10, batch_size=128)
    bc_path = f"{save_dir}/bc_policy.pt"
    bc_trainer.save(bc_path)
    print(f"  BC final accuracy: {bc_metrics['accuracy'][-1]:.3f}")

    # Stage 2: PPO Fine-tuning
    print("\n[Stage 2] PPO fine-tuning with Guardian Drive reward...")
    ppo_trainer = PPOTrainer(policy, device=device, n_steps=256)
    ppo_metrics = ppo_trainer.train(env, n_updates=20)
    ppo_path = f"{save_dir}/ppo_policy.pt"
    ppo_trainer.save(ppo_path)

    # Evaluation
    print("\n[Evaluation] BC vs PPO vs Expert...")
    evaluator = EvaluationRunner(env, n_episodes=5, max_steps_per_episode=200)
    results = evaluator.compare_bc_vs_ppo(
        bc_path, ppo_path,
        save_path=f"{save_dir}/eval_results.json"
    )

    print(f"\n[Project 1 Summary]")
    print(f"  BC  safety accuracy: {results['bc']['safety_accuracy_mean']:.3f}")
    print(f"  PPO safety accuracy: {results['ppo']['safety_accuracy_mean']:.3f}")
    print(f"  Expert (upper bound): {results['expert']['safety_accuracy_mean']:.3f}")
    return results


# ─────────────────────────────────────────────
# Project 2: Fleet Telemetry Pipeline
# ─────────────────────────────────────────────

def run_fleet_project(save_dir: str = "outputs/fleet_telemetry"):
    print("\n" + "=" * 60)
    print("PROJECT 2: Fleet Telemetry Pipeline")
    print("nuPlan + Waymo + Guardian Pi → Parquet → DuckDB → Rare Events")
    print("=" * 60)

    from fleet_telemetry.ingestion.log_ingestor import (
        FleetTelemetryIngestor,
        NuPlanLogParser,
        WaymoLogParser,
        GuardianPiLogParser,
    )
    from fleet_telemetry.query.rare_event_miner import RareEventMiner

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    parquet_dir = f"{save_dir}/parquet"

    ingestor = FleetTelemetryIngestor(output_dir=parquet_dir, batch_size=500)

    # Ingest all three sources
    sources = [
        ("nuPlan (1300h real driving)", NuPlanLogParser(synthetic_n=1000)),
        ("Waymo Open Dataset", WaymoLogParser(synthetic_n=500)),
        ("Guardian Drive Pi logs", GuardianPiLogParser(synthetic_n=300)),
    ]

    total_stats = {}
    for name, parser in sources:
        print(f"\n  Ingesting {name}...")
        t0 = time.time()
        stats = ingestor.ingest(parser.parse())
        elapsed = time.time() - t0
        print(f"    Events: {stats['total_ingested']} | "
              f"Rejected: {stats['rejected']} | "
              f"Repaired: {stats['repaired']} | "
              f"Rate: {stats['total_ingested'] / elapsed:.0f} events/sec")
        total_stats[name] = stats

    # Rare event mining
    print("\n  Mining rare events...")
    miner = RareEventMiner(parquet_dir=parquet_dir)

    crash_precursors = miner.mine_crash_precursors(g_threshold=1.5)
    drowsy_seqs = miner.mine_drowsiness_sequences(perclos_threshold=0.25)
    cardiac_events = miner.mine_cardiac_events()
    fleet_stats = miner.compute_fleet_statistics()

    print(f"\n  Fleet statistics:")
    for k, v in fleet_stats.items():
        if k != "error":
            print(f"    {k}: {v}")

    print(f"\n  Rare events found:")
    print(f"    Crash precursors (g>1.5): {len(crash_precursors)}")
    print(f"    Drowsiness sequences: {len(drowsy_seqs)}")
    print(f"    Cardiac events: {len(cardiac_events)}")

    # Export training datasets
    miner.export_training_dataset("crash_precursors", f"{save_dir}/datasets/crash_precursors.json")
    miner.export_training_dataset("drowsiness", f"{save_dir}/datasets/drowsiness_sequences.json")
    miner.export_training_dataset("cardiac", f"{save_dir}/datasets/cardiac_events.json")

    # Save summary
    summary = {
        "total_events": sum(s["total_ingested"] for s in total_stats.values()),
        "sources": list(total_stats.keys()),
        "crash_precursors": len(crash_precursors),
        "drowsy_sequences": len(drowsy_seqs),
        "cardiac_events": len(cardiac_events),
        "fleet_stats": fleet_stats,
    }
    with open(f"{save_dir}/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[Project 2 Summary]")
    print(f"  Total events ingested: {summary['total_events']}")
    print(f"  Rare events mined: {sum([len(crash_precursors), len(drowsy_seqs), len(cardiac_events)])}")
    return summary


# ─────────────────────────────────────────────
# Project 3: BEV Perception
# ─────────────────────────────────────────────

def run_bev_project(save_dir: str = "outputs/bev_perception", device: str = "cpu"):
    print("\n" + "=" * 60)
    print("PROJECT 3: BEVFormer Perception Integration")
    print("Multi-camera → BEV features → 3D detection → trajectory risk")
    print("=" * 60)

    from bev_perception.model.bev_perception import (
        GuardianBEVPerception, NuScenesEvaluator
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model = GuardianBEVPerception(n_cameras=6)
    model.eval()

    print("\n  Running synthetic forward passes (simulates 6-camera input)...")
    rng = np.random.default_rng(42)
    results_list = []

    for i in range(10):
        n_objects = int(rng.integers(0, 15))
        ego_speed = float(rng.uniform(0, 30))
        with torch.no_grad():
            result = model.forward_synthetic(
                ego_speed_mps=ego_speed,
                n_objects=n_objects,
                rng=rng,
            )
        results_list.append({
            "run": i,
            "n_objects_in": n_objects,
            "n_detections": result["n_detections"],
            "trajectory_risk": result["trajectory_risk"],
            "n_danger_objects": result["n_danger_objects"],
            "min_ttc_sec": result["min_ttc_sec"],
            "closest_object_m": result["closest_object_m"],
        })
        print(f"    Run {i+1}: objects={n_objects} detected={result['n_detections']} "
              f"risk={result['trajectory_risk']:.3f} TTC={result['min_ttc_sec']:.1f}s")

    # nuScenes evaluation
    print("\n  nuScenes detection evaluation...")
    evaluator = NuScenesEvaluator()
    nds_results = evaluator.compute_nds([], [])
    print(f"    NDS: {nds_results['NDS']:.3f}")
    print(f"    mAP: {nds_results['mAP']:.3f}")
    print(f"    Note: {nds_results['note']}")

    # Show how BEV risk integrates into Guardian Drive fusion
    print("\n  Guardian Drive fusion integration:")
    print("    r = 0.40×r_phys + 0.30×r_neuro + 0.20×r_imu + 0.10×r_ctx")
    print("    r_ctx now includes: trajectory_risk (BEVFormer) + temp + steering + speech + GSR")
    avg_risk = np.mean([r["trajectory_risk"] for r in results_list])
    print(f"    Average trajectory_risk from BEV: {avg_risk:.3f}")

    # Save results
    summary = {
        "model_params": sum(p.numel() for p in model.parameters()),
        "nds_results": nds_results,
        "forward_pass_results": results_list,
        "bev_grid_size": f"{GuardianBEVPerception.__init__.__doc__ or '50×50 @ 2m'}",
        "integration": "trajectory_risk feeds r_ctx in Guardian Drive fusion",
    }
    with open(f"{save_dir}/bev_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[Project 3 Summary]")
    print(f"  Model parameters: {summary['model_params']:,}")
    print(f"  NDS (nuScenes): {nds_results['NDS']:.3f}")
    print(f"  Average BEV trajectory risk: {avg_risk:.3f}")
    return summary


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Guardian Drive v2 Training")
    parser.add_argument("--project", choices=["all", "carla", "fleet", "bev"],
                        default="all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("Guardian Drive v2.0 — Complete Build")
    print("Akilan Manivannan & Akila Lourdes Miriyala Francis")
    print("LIU Brooklyn — MS Artificial Intelligence")
    print("=" * 60)
    print(f"Device: {args.device} | Project: {args.project}")

    t_start = time.time()
    all_results = {}

    if args.project in ("all", "carla"):
        all_results["carla"] = run_carla_project(
            f"{args.output_dir}/carla_agent", args.device)

    if args.project in ("all", "fleet"):
        all_results["fleet"] = run_fleet_project(
            f"{args.output_dir}/fleet_telemetry")

    if args.project in ("all", "bev"):
        all_results["bev"] = run_bev_project(
            f"{args.output_dir}/bev_perception", args.device)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"All projects complete in {elapsed:.1f}s")
    print(f"Results saved to {args.output_dir}/")

    with open(f"{args.output_dir}/master_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
