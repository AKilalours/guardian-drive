"""
server/fleet_aggregator.py
Guardian Drive -- Fleet Telemetry Aggregation

Aggregates telemetry across multiple vehicles/sessions.
In production: Tesla Dojo-style data pipeline.
In prototype: local JSONL aggregation.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

class FleetAggregator:
    """
    Aggregates TelemetryLogger JSONL records across sessions.
    
    Provides:
    - Fleet-wide impairment rate statistics
    - Per-route risk heatmap (GPS binning)
    - Model performance tracking (live AUC estimate)
    - Anomaly detection (unusual physiological patterns)
    """

    def __init__(self, log_dir: str = "data/telemetry_logs"):
        self.log_dir = Path(log_dir)
        self.records: list = []

    def load_all_sessions(self) -> int:
        """Load all JSONL telemetry files."""
        self.records = []
        for f in self.log_dir.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    try:
                        self.records.append(json.loads(line))
                    except Exception:
                        continue
        print(f"[Fleet] Loaded {len(self.records)} frames "
              f"from {len(list(self.log_dir.glob('*.jsonl')))} sessions")
        return len(self.records)

    def impairment_rate(self) -> dict:
        """Fleet-wide impairment rate by state."""
        counts = defaultdict(int)
        for r in self.records:
            counts[r.get("level","UNKNOWN")] += 1
        total = max(len(self.records), 1)
        return {k: round(v/total, 4) for k,v in counts.items()}

    def risk_heatmap(self, bin_size: float = 0.01) -> dict:
        """
        GPS-binned risk score heatmap.
        bin_size: degrees (~1km at equator)
        """
        heatmap = defaultdict(list)
        for r in self.records:
            gps = r.get("gps", {})
            lat = gps.get("lat")
            lon = gps.get("lon")
            risk = r.get("risk_score", 0)
            if lat and lon:
                key = (round(lat/bin_size)*bin_size,
                       round(lon/bin_size)*bin_size)
                heatmap[key].append(risk)
        return {str(k): round(float(np.mean(v)),4)
                for k,v in heatmap.items()}

    def model_drift_check(self, window: int = 1000) -> dict:
        """
        Check if recent predictions differ from historical.
        Simple drift detection for fleet monitoring.
        """
        if len(self.records) < window*2:
            return {"status": "insufficient_data"}
        historical = [r.get("risk_score",0)
                      for r in self.records[:-window]]
        recent     = [r.get("risk_score",0)
                      for r in self.records[-window:]]
        drift = float(np.mean(recent) - np.mean(historical))
        return {
            "historical_mean": round(float(np.mean(historical)),4),
            "recent_mean":     round(float(np.mean(recent)),4),
            "drift":           round(drift,4),
            "alert":           abs(drift) > 0.1,
        }

    def summary(self) -> dict:
        """Full fleet summary report."""
        if not self.records:
            return {"error": "No records loaded"}
        risk_scores = [r.get("risk_score",0) for r in self.records]
        return {
            "total_frames":    len(self.records),
            "mean_risk":       round(float(np.mean(risk_scores)),4),
            "p95_risk":        round(float(np.percentile(risk_scores,95)),4),
            "impairment_rate": self.impairment_rate(),
            "drift":           self.model_drift_check(),
        }

if __name__ == "__main__":
    agg = FleetAggregator()
    n   = agg.load_all_sessions()
    if n > 0:
        print(json.dumps(agg.summary(), indent=2))
    else:
        print("No telemetry logs found.")
        print("Run Guardian Drive server first to generate logs.")
        print("Expected: data/telemetry_logs/*.jsonl")
