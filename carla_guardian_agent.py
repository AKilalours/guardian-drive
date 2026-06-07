"""Guardian Drive CARLA Agent"""
import carla, time, math, json
from pathlib import Path

class GuardianDriveAgent:
    def __init__(self):
        self.interventions = 0
    def compute_control(self, risk, state):
        import carla as c
        ctrl = c.VehicleControl()
        if risk > 0.85:
            ctrl.throttle=0.0; ctrl.brake=1.0
            self.interventions+=1
        elif risk > 0.65:
            ctrl.throttle=0.0; ctrl.brake=0.6
            self.interventions+=1
        elif risk > 0.40:
            ctrl.throttle=0.2; ctrl.brake=0.3
        else:
            ctrl.throttle=0.5; ctrl.brake=0.0
        return ctrl

if __name__ == '__main__':
    print("Guardian Drive CARLA Agent ready")
