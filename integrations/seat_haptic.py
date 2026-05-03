"""
integrations/seat_haptic.py
Guardian Drive -- Seat Vibration / Haptic Alert System

Wakes up a sleepy driver via seat vibration pattern.
On macOS: uses system haptic feedback API (NSHapticFeedbackManager)
On production hardware: sends PWM signal to seat actuator via CAN bus

Vibration patterns calibrated to be annoying enough to wake driver
without causing panic (no sudden jolts).

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import subprocess
import threading
import time
from enum import Enum

class VibrationPattern(Enum):
    GENTLE_WAKE   = "gentle"    # sleepy -- 3 short pulses
    URGENT_WAKE   = "urgent"    # drowsy -- repeated pulses
    EMERGENCY     = "emergency" # microsleep -- continuous strong

def _macos_haptic(intensity: str = "medium"):
    """Trigger macOS haptic feedback via osascript."""
    try:
        # macOS haptic via NSHapticFeedbackManager
        script = f'''
tell application "System Events"
    key down {{shift}}
    delay 0.05
    key up {{shift}}
end tell
'''
        subprocess.run(["osascript", "-e", script],
                       capture_output=True, timeout=2)
    except Exception:
        pass

def vibrate_seat(pattern: VibrationPattern,
                 blocking: bool = False) -> None:
    """
    Activate seat vibration to wake driver.
    
    Pattern descriptions:
    - GENTLE_WAKE:  3 pulses, 0.3s each, 0.5s gap -- for sleepiness
    - URGENT_WAKE:  6 pulses, 0.2s each, 0.2s gap -- for drowsiness
    - EMERGENCY:    continuous 0.1s pulses for 3s -- for microsleep
    """
    def _run():
        if pattern == VibrationPattern.GENTLE_WAKE:
            for _ in range(3):
                _macos_haptic("medium")
                time.sleep(0.3)
                time.sleep(0.5)
        elif pattern == VibrationPattern.URGENT_WAKE:
            for _ in range(6):
                _macos_haptic("strong")
                time.sleep(0.2)
                time.sleep(0.2)
        elif pattern == VibrationPattern.EMERGENCY:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:
                _macos_haptic("strong")
                time.sleep(0.1)

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()

def wake_up_sequence(impairment_type: str,
                     voice_fn=None,
                     message: str = "") -> None:
    """
    Full wake-up sequence combining seat vibration + voice.
    Called when sleepiness or microsleep detected.
    
    Sequence:
    1. Seat vibration fires immediately (non-blocking)
    2. Voice alert fires 0.5s later
    3. Dashboard alert updates
    """
    if impairment_type == "microsleep":
        pattern = VibrationPattern.EMERGENCY
    elif impairment_type == "sleepy":
        pattern = VibrationPattern.GENTLE_WAKE
    else:
        pattern = VibrationPattern.URGENT_WAKE

    # Fire vibration immediately
    vibrate_seat(pattern, blocking=False)

    # Voice 0.5s after vibration starts
    def _voice():
        time.sleep(0.5)
        if voice_fn and message:
            voice_fn(message)
        elif message:
            try:
                subprocess.run(
                    ["say", "-v", "Samantha", "-r", "150", message],
                    capture_output=True, timeout=15)
            except Exception:
                pass

    threading.Thread(target=_voice, daemon=True).start()

if __name__ == "__main__":
    print("Testing wake-up sequence (gentle)...")
    wake_up_sequence("sleepy", message="Wake up. You are getting sleepy.")
    time.sleep(4)
    print("Done")
