"""
Stub for learned Task B inference.
Original learned/ folder was removed during cleanup.
Falls back to WESAD rule-based detection.
"""
class TaskBInference:
    def __init__(self):
        self._available = False

    def predict(self, feature_vector):
        """Return None — triggers fallback to rule-based detection."""
        return None
