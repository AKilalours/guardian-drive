"""
integrations/impairment_classifier.py
Guardian Drive -- Drowsiness vs Fatigue vs Sleepiness Classifier

Three distinct impairment states with different physiological signatures
and different intervention strategies:

SLEEPINESS   -- acute urge to sleep, PERCLOS primary signal
               Intervention: immediate wake-up vibration + coffee shop
               
DROWSINESS   -- reduced vigilance, early impairment
               Intervention: voice alert + rest stop suggestion
               
FATIGUE      -- sustained physiological depletion, HRV primary signal  
               Intervention: mandatory rest, do not resume driving

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ImpairmentType(Enum):
    ALERT      = "alert"
    SLEEPY     = "sleepy"       # acute -- PERCLOS driven
    DROWSY     = "drowsy"       # moderate -- physio + camera
    FATIGUED   = "fatigued"     # sustained -- HRV driven
    MICROSLEEP = "microsleep"   # critical -- PERCLOS >80%

@dataclass
class ImpairmentResult:
    state:           ImpairmentType
    confidence:      float
    perclos_30s:     float
    ear_current:     float
    yawn_count:      int
    hrv_rmssd:       float
    blink_rate:      float
    sustained_mins:  float
    primary_signal:  str
    intervention:    str
    wake_up_needed:  bool
    poi_type:        str          # what POI to search for
    voice_message:   str

class ImpairmentClassifier:
    """
    Classifies driver impairment into three distinct categories
    with different physiological signatures and interventions.
    
    Key distinctions:
    - Sleepiness: PERCLOS > 0.25, yawns > 3/30s, EAR declining
    - Drowsiness: low-arousal TCN score + mild PERCLOS + HRV mild drop
    - Fatigue:    sustained HRV depression (RMSSD < 20ms), >45min driving
    - Microsleep: PERCLOS > 0.80 for >1s -- EMERGENCY
    """
    
    # Thresholds calibrated from literature
    PERCLOS_SLEEPY     = 0.25   # Wierwille & Ellsworth 1994
    PERCLOS_DROWSY     = 0.15
    PERCLOS_MICROSLEEP = 0.80   # eyes closed >80% = microsleep
    
    EAR_CLOSED         = 0.18   # calibrated per subject
    YAWN_SLEEPY        = 3      # yawns per 30s
    YAWN_DROWSY        = 1
    
    HRV_FATIGUE        = 20.0   # RMSSD < 20ms = fatigue (Thayer et al.)
    HRV_DROWSY         = 30.0   # RMSSD < 30ms = early fatigue
    
    BLINK_SLEEPY       = 25     # blinks/min > 25 = sleepy
    BLINK_LOW          = 8      # blinks/min < 8 = microsleep risk
    
    FATIGUE_DRIVE_MINS = 45     # >45 min continuous = fatigue risk

    def classify(self,
                 perclos_30s: float,
                 ear: float,
                 yawn_count: int,
                 hrv_rmssd: float,
                 blink_rate: float,
                 tcn_prob: float,
                 drive_minutes: float = 0.0) -> ImpairmentResult:
        """
        Classify impairment state from multimodal features.
        Priority: MICROSLEEP > SLEEPY > FATIGUED > DROWSY > ALERT
        """
        
        # ── Microsleep detection (highest priority) ───────────────
        if perclos_30s > self.PERCLOS_MICROSLEEP:
            return ImpairmentResult(
                state           = ImpairmentType.MICROSLEEP,
                confidence      = min(1.0, perclos_30s),
                perclos_30s     = perclos_30s,
                ear_current     = ear,
                yawn_count      = yawn_count,
                hrv_rmssd       = hrv_rmssd,
                blink_rate      = blink_rate,
                sustained_mins  = drive_minutes,
                primary_signal  = "PERCLOS >80% -- eyes closed",
                intervention    = "EMERGENCY -- microsleep detected",
                wake_up_needed  = True,
                poi_type        = "hospital",
                voice_message   = "MICROSLEEP DETECTED. Pull over immediately. Activating emergency protocol."
            )
        
        # ── Sleepiness (acute -- camera primary) ─────────────────
        sleepy_score = (
            (perclos_30s / self.PERCLOS_SLEEPY) * 0.40 +
            (yawn_count  / self.YAWN_SLEEPY)    * 0.35 +
            (max(0, blink_rate - self.BLINK_SLEEPY) / 10) * 0.25
        )
        
        if sleepy_score > 0.80 or (perclos_30s > self.PERCLOS_SLEEPY
                                    and yawn_count >= self.YAWN_SLEEPY):
            return ImpairmentResult(
                state           = ImpairmentType.SLEEPY,
                confidence      = min(1.0, sleepy_score),
                perclos_30s     = perclos_30s,
                ear_current     = ear,
                yawn_count      = yawn_count,
                hrv_rmssd       = hrv_rmssd,
                blink_rate      = blink_rate,
                sustained_mins  = drive_minutes,
                primary_signal  = f"PERCLOS={perclos_30s:.2f} Yawns={yawn_count}",
                intervention    = "Wake up -- seat vibration + voice + coffee shop",
                wake_up_needed  = True,
                poi_type        = "cafe",
                voice_message   = (
                    "You are getting sleepy. Your eyes are closing. "
                    "There is a coffee shop nearby. Pull over and take a break.")
            )
        
        # ── Fatigue (sustained -- HRV primary) ────────────────────
        fatigue_score = 0.0
        if hrv_rmssd < self.HRV_FATIGUE:
            fatigue_score += 0.50
        elif hrv_rmssd < self.HRV_DROWSY:
            fatigue_score += 0.25
        if drive_minutes > self.FATIGUE_DRIVE_MINS:
            fatigue_score += min(0.50,
                (drive_minutes - self.FATIGUE_DRIVE_MINS) / 60.0)
        
        if fatigue_score > 0.60:
            return ImpairmentResult(
                state           = ImpairmentType.FATIGUED,
                confidence      = min(1.0, fatigue_score),
                perclos_30s     = perclos_30s,
                ear_current     = ear,
                yawn_count      = yawn_count,
                hrv_rmssd       = hrv_rmssd,
                blink_rate      = blink_rate,
                sustained_mins  = drive_minutes,
                primary_signal  = f"RMSSD={hrv_rmssd:.1f}ms Drive={drive_minutes:.0f}min",
                intervention    = "Mandatory rest -- do not resume driving",
                wake_up_needed  = False,
                poi_type        = "motel",
                voice_message   = (
                    "You have been driving for a long time. "
                    "Your heart rate variability indicates physical fatigue. "
                    "There is a rest stop nearby. You must stop driving.")
            )
        
        # ── Drowsiness (moderate -- TCN + mild camera) ────────────
        drowsy_score = (
            tcn_prob * 0.50 +
            (perclos_30s / self.PERCLOS_DROWSY) * 0.30 +
            (yawn_count  / max(1, self.YAWN_DROWSY)) * 0.20
        )
        
        if drowsy_score > 0.50:
            return ImpairmentResult(
                state           = ImpairmentType.DROWSY,
                confidence      = min(1.0, drowsy_score),
                perclos_30s     = perclos_30s,
                ear_current     = ear,
                yawn_count      = yawn_count,
                hrv_rmssd       = hrv_rmssd,
                blink_rate      = blink_rate,
                sustained_mins  = drive_minutes,
                primary_signal  = f"TCN={tcn_prob:.3f} PERCLOS={perclos_30s:.2f}",
                intervention    = "Voice alert + rest stop suggestion",
                wake_up_needed  = False,
                poi_type        = "rest_area",
                voice_message   = (
                    "Early drowsiness detected. "
                    "Plan a rest break within the next 20 minutes.")
            )
        
        # ── Alert ─────────────────────────────────────────────────
        return ImpairmentResult(
            state           = ImpairmentType.ALERT,
            confidence      = 1.0 - max(sleepy_score, fatigue_score, drowsy_score),
            perclos_30s     = perclos_30s,
            ear_current     = ear,
            yawn_count      = yawn_count,
            hrv_rmssd       = hrv_rmssd,
            blink_rate      = blink_rate,
            sustained_mins  = drive_minutes,
            primary_signal  = "All signals nominal",
            intervention    = "None",
            wake_up_needed  = False,
            poi_type        = "none",
            voice_message   = ""
        )

if __name__ == "__main__":
    clf = ImpairmentClassifier()
    
    print("=== Impairment Classification Demo ===\n")
    
    tests = [
        ("Alert driver",
         dict(perclos_30s=0.05, ear=0.28, yawn_count=0,
              hrv_rmssd=45.0, blink_rate=15, tcn_prob=0.10,
              drive_minutes=20)),
        ("Sleepy driver",
         dict(perclos_30s=0.35, ear=0.16, yawn_count=4,
              hrv_rmssd=38.0, blink_rate=28, tcn_prob=0.45,
              drive_minutes=30)),
        ("Fatigued driver",
         dict(perclos_30s=0.10, ear=0.22, yawn_count=1,
              hrv_rmssd=15.0, blink_rate=12, tcn_prob=0.35,
              drive_minutes=90)),
        ("Drowsy driver",
         dict(perclos_30s=0.18, ear=0.19, yawn_count=2,
              hrv_rmssd=28.0, blink_rate=18, tcn_prob=0.60,
              drive_minutes=25)),
        ("Microsleep",
         dict(perclos_30s=0.85, ear=0.08, yawn_count=5,
              hrv_rmssd=20.0, blink_rate=5, tcn_prob=0.90,
              drive_minutes=60)),
    ]
    
    for name, params in tests:
        result = clf.classify(**params)
        print(f"Scenario: {name}")
        print(f"  State:         {result.state.value.upper()}")
        print(f"  Confidence:    {result.confidence:.3f}")
        print(f"  Primary:       {result.primary_signal}")
        print(f"  Intervention:  {result.intervention}")
        print(f"  Wake up:       {result.wake_up_needed}")
        print(f"  POI type:      {result.poi_type}")
        print(f"  Voice:         {result.voice_message[:60]}...")
        print()
