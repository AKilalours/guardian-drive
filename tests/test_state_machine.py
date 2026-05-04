"""Unit tests for 5-state safety Mealy machine."""
import pytest

def state_machine(r, thresh):
    if r < thresh:           return "NOMINAL"
    elif r < thresh + 0.20:  return "ADVISORY"
    elif r < thresh + 0.40:  return "CAUTION"
    elif r < thresh + 0.60:  return "PULLOVER"
    else:                    return "ESCALATE"

def test_nominal():
    assert state_machine(0.10, 0.35) == "NOMINAL"

def test_advisory():
    assert state_machine(0.40, 0.35) == "ADVISORY"

def test_caution():
    assert state_machine(0.60, 0.35) == "CAUTION"

def test_pullover():
    assert state_machine(0.80, 0.35) == "PULLOVER"

def test_escalate():
    assert state_machine(1.00, 0.35) == "ESCALATE"

def test_threshold_shift():
    # Lower threshold (more traffic) = earlier escalation
    assert state_machine(0.28, 0.25) == "ADVISORY"
    assert state_machine(0.28, 0.35) == "NOMINAL"

def test_boundary_nominal():
    assert state_machine(0.349, 0.35) == "NOMINAL"

def test_boundary_advisory():
    assert state_machine(0.350, 0.35) == "ADVISORY"

def test_all_states_reachable():
    states = {state_machine(r, 0.35)
              for r in [0.1, 0.4, 0.6, 0.8, 1.0]}
    assert states == {"NOMINAL","ADVISORY","CAUTION",
                      "PULLOVER","ESCALATE"}

if __name__ == "__main__":
    test_nominal(); test_advisory(); test_caution()
    test_pullover(); test_escalate(); test_threshold_shift()
    test_boundary_nominal(); test_boundary_advisory()
    test_all_states_reachable()
    print("All state machine tests passed")
