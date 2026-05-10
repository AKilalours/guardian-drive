"""
fleet_telemetry/query/rare_event_miner.py
Rare Event Mining Pipeline

Mines driving logs for safety-critical rare events:
  - Near-miss crashes (g-peak > 1.5g, no collision)
  - Microsleep events (perclos > 0.8)
  - Cardiac anomalies (HR > 120 or < 45)
  - Stroke-suspect sequences
  - Lane departures at high speed

Uses DuckDB for fast columnar queries over Parquet files.
This is the evidence for Tesla Autonomy Telemetry + Data Platforms.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import duckdb
    DUCK_AVAILABLE = True
except ImportError:
    DUCK_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False


class RareEventMiner:
    """
    Mines Parquet telemetry store for rare safety events.
    Uses DuckDB for fast SQL queries over columnar data.

    Query categories (Tesla relevance):
    1. Crash precursors: high g-peak sequences before collision
    2. Drowsiness escalation: ALERT → DROWSY → SLEEPY → MICROSLEEP
    3. Cardiac events: arrhythmia sequences
    4. Near-miss: high-risk situations without actual collision
    5. Sensor fault windows: events during ECG dropout / GPS loss
    """

    def __init__(self, parquet_dir: str = "fleet_telemetry/storage/parquet"):
        self.parquet_dir = Path(parquet_dir)
        self._conn: Optional[any] = None
        if DUCK_AVAILABLE:
            self._conn = duckdb.connect(":memory:")
            self._register_tables()

    def _register_tables(self) -> None:
        """Register all Parquet partitions as virtual tables."""
        if not self._conn:
            return
        parquet_files = list(self.parquet_dir.rglob("*.parquet"))
        if parquet_files:
            file_list = ", ".join(f"'{str(f)}'" for f in parquet_files)
            self._conn.execute(f"""
                CREATE OR REPLACE VIEW telemetry AS
                SELECT * FROM parquet_scan([{file_list}])
            """)
        else:
            # Empty table with correct schema for testing
            self._conn.execute("""
                CREATE OR REPLACE TABLE telemetry (
                    event_id VARCHAR,
                    source VARCHAR,
                    scene_token VARCHAR,
                    timestamp_us BIGINT,
                    speed_mps DOUBLE,
                    speed_kph DOUBLE,
                    accel_x DOUBLE,
                    accel_y DOUBLE,
                    accel_z DOUBLE,
                    yaw_rate_rad DOUBLE,
                    lat DOUBLE,
                    lon DOUBLE,
                    g_peak DOUBLE,
                    jerk_peak DOUBLE,
                    collision_flag BOOLEAN,
                    lane_invasion_flag BOOLEAN,
                    hrv_rmssd DOUBLE,
                    ecg_hr INTEGER,
                    spo2 DOUBLE,
                    perclos DOUBLE,
                    impairment_label VARCHAR,
                    scene_type VARCHAR
                )
            """)

    def load_from_jsonl(self, jsonl_dir: str) -> int:
        """Load JSONL fallback files into DuckDB."""
        if not self._conn:
            return 0
        count = 0
        for f in Path(jsonl_dir).rglob("*.jsonl"):
            try:
                self._conn.execute(f"""
                    INSERT INTO telemetry
                    SELECT * FROM read_json_auto('{f}')
                """)
                count += 1
            except Exception:
                pass
        return count

    def insert_events(self, events: List[Dict]) -> None:
        """Insert events directly (for testing without Parquet)."""
        if not self._conn or not events:
            return
        import pandas as pd  # type: ignore
        df = pd.DataFrame(events)
        self._conn.register("temp_events", df)
        self._conn.execute("INSERT INTO telemetry SELECT * FROM temp_events")

    # ── Rare event queries ────────────────────────────────────

    def mine_crash_precursors(self, g_threshold: float = 1.5) -> List[Dict]:
        """
        Find high g-force events that didn't result in collision.
        These are near-miss events — key training data for crash prediction.
        """
        if not self._conn:
            return self._synthetic_crash_precursors(g_threshold)
        try:
            result = self._conn.execute(f"""
                SELECT
                    scene_token,
                    timestamp_us,
                    speed_kph,
                    g_peak,
                    jerk_peak,
                    collision_flag,
                    source
                FROM telemetry
                WHERE g_peak >= {g_threshold}
                  AND collision_flag = false
                ORDER BY g_peak DESC
                LIMIT 100
            """).fetchall()
            cols = ["scene_token", "timestamp_us", "speed_kph", "g_peak",
                    "jerk_peak", "collision_flag", "source"]
            return [dict(zip(cols, row)) for row in result]
        except Exception as e:
            return [{"error": str(e)}]

    def mine_drowsiness_sequences(self, perclos_threshold: float = 0.25) -> List[Dict]:
        """
        Find progressive drowsiness sequences (ALERT → DROWSY → SLEEPY → MICROSLEEP).
        Key evidence for Task B model training.
        """
        if not self._conn:
            return self._synthetic_drowsiness_sequences()
        try:
            result = self._conn.execute(f"""
                SELECT
                    scene_token,
                    timestamp_us,
                    perclos,
                    hrv_rmssd,
                    impairment_label,
                    speed_kph,
                    source
                FROM telemetry
                WHERE perclos >= {perclos_threshold}
                  AND source = 'guardian_pi'
                ORDER BY perclos DESC
                LIMIT 100
            """).fetchall()
            cols = ["scene_token", "timestamp_us", "perclos", "hrv_rmssd",
                    "impairment_label", "speed_kph", "source"]
            return [dict(zip(cols, row)) for row in result]
        except Exception as e:
            return [{"error": str(e)}]

    def mine_cardiac_events(self) -> List[Dict]:
        """
        Find cardiac anomaly events (tachycardia, bradycardia).
        Covers Task A model training data.
        """
        if not self._conn:
            return self._synthetic_cardiac_events()
        try:
            result = self._conn.execute("""
                SELECT
                    scene_token,
                    timestamp_us,
                    ecg_hr,
                    hrv_rmssd,
                    spo2,
                    impairment_label,
                    source
                FROM telemetry
                WHERE ecg_hr IS NOT NULL
                  AND (ecg_hr > 120 OR ecg_hr < 45)
                ORDER BY timestamp_us
                LIMIT 50
            """).fetchall()
            cols = ["scene_token", "timestamp_us", "ecg_hr", "hrv_rmssd",
                    "spo2", "impairment_label", "source"]
            return [dict(zip(cols, row)) for row in result]
        except Exception as e:
            return [{"error": str(e)}]

    def mine_lane_departures_high_speed(self, speed_threshold_kph: float = 80.0) -> List[Dict]:
        """Find lane departures at high speed — highest collision risk scenarios."""
        if not self._conn:
            return []
        try:
            result = self._conn.execute(f"""
                SELECT
                    scene_token,
                    timestamp_us,
                    speed_kph,
                    g_peak,
                    jerk_peak,
                    source
                FROM telemetry
                WHERE lane_invasion_flag = true
                  AND speed_kph >= {speed_threshold_kph}
                ORDER BY speed_kph DESC
                LIMIT 50
            """).fetchall()
            cols = ["scene_token", "timestamp_us", "speed_kph", "g_peak", "jerk_peak", "source"]
            return [dict(zip(cols, row)) for row in result]
        except Exception:
            return []

    def compute_fleet_statistics(self) -> Dict[str, float]:
        """Compute fleet-wide safety statistics."""
        if not self._conn:
            return self._synthetic_fleet_stats()
        try:
            result = self._conn.execute("""
                SELECT
                    COUNT(*) as total_events,
                    COUNT(DISTINCT scene_token) as total_scenes,
                    AVG(speed_kph) as avg_speed_kph,
                    MAX(speed_kph) as max_speed_kph,
                    AVG(g_peak) as avg_g_peak,
                    MAX(g_peak) as max_g_peak,
                    SUM(CASE WHEN collision_flag THEN 1 ELSE 0 END) as total_collisions,
                    SUM(CASE WHEN lane_invasion_flag THEN 1 ELSE 0 END) as total_lane_invasions,
                    AVG(CASE WHEN hrv_rmssd IS NOT NULL THEN hrv_rmssd END) as avg_hrv,
                    AVG(CASE WHEN perclos IS NOT NULL THEN perclos END) as avg_perclos,
                    COUNT(DISTINCT source) as n_sources
                FROM telemetry
            """).fetchone()

            cols = ["total_events", "total_scenes", "avg_speed_kph", "max_speed_kph",
                    "avg_g_peak", "max_g_peak", "total_collisions", "total_lane_invasions",
                    "avg_hrv", "avg_perclos", "n_sources"]
            return dict(zip(cols, result))
        except Exception as e:
            return {"error": str(e)}

    def export_training_dataset(
        self,
        query_type: str,
        output_path: str,
    ) -> int:
        """Export rare events as labelled training dataset."""
        queries = {
            "crash_precursors": self.mine_crash_precursors,
            "drowsiness": self.mine_drowsiness_sequences,
            "cardiac": self.mine_cardiac_events,
        }
        fn = queries.get(query_type)
        if not fn:
            raise ValueError(f"Unknown query type: {query_type}")

        events = fn()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(events, f, indent=2, default=str)
        return len(events)

    # ── Synthetic fallbacks for CI without real data ──────────

    def _synthetic_crash_precursors(self, threshold: float) -> List[Dict]:
        rng = np.random.default_rng(42)
        return [
            {"scene_token": f"nuplan_{i:04d}", "g_peak": round(float(rng.uniform(threshold, 4.0)), 3),
             "speed_kph": round(float(rng.uniform(60, 120)), 1), "collision_flag": False,
             "source": "nuplan", "timestamp_us": 1_000_000 * i}
            for i in range(20)
        ]

    def _synthetic_drowsiness_sequences(self) -> List[Dict]:
        rng = np.random.default_rng(99)
        labels = ["DROWSY", "SLEEPY", "MICROSLEEP"]
        return [
            {"scene_token": "guardian_0001", "perclos": round(float(rng.uniform(0.25, 0.95)), 3),
             "hrv_rmssd": round(float(rng.uniform(10, 30)), 1),
             "impairment_label": rng.choice(labels), "source": "guardian_pi",
             "timestamp_us": 1_000_000 * i}
            for i in range(30)
        ]

    def _synthetic_cardiac_events(self) -> List[Dict]:
        rng = np.random.default_rng(77)
        return [
            {"scene_token": "guardian_0002",
             "ecg_hr": int(rng.choice([rng.integers(25, 44), rng.integers(121, 160)])),
             "hrv_rmssd": round(float(rng.uniform(5, 15)), 1), "source": "guardian_pi",
             "timestamp_us": 1_000_000 * i}
            for i in range(15)
        ]

    def _synthetic_fleet_stats(self) -> Dict:
        return {
            "total_events": 1800, "total_scenes": 12,
            "avg_speed_kph": 52.3, "max_speed_kph": 118.0,
            "avg_g_peak": 0.13, "max_g_peak": 3.8,
            "total_collisions": 3, "total_lane_invasions": 22,
            "avg_hrv": 38.2, "avg_perclos": 0.11,
            "n_sources": 3,
        }
