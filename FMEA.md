# Guardian Drive -- Failure Mode and Effects Analysis (FMEA)

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
Research prototype -- not a safety-critical system

## Scope
This FMEA covers the Guardian Drive prototype pipeline.
It is NOT a regulatory-grade FMEA. It demonstrates understanding
of safety engineering methodology for the Tesla application.

## Failure Modes

| ID | Component | Failure Mode | Effect | Severity | Likelihood | RPN | Mitigation |
|----|-----------|-------------|--------|----------|-----------|-----|------------|
| F01 | ECG sensor | Electrode detachment | SQI drops, abstain | High | Medium | 12 | SQI gating abstains |
| F02 | Task B TCN | False NOMINAL (miss drowsiness) | Driver not alerted | Critical | Low | 9 | Camera EAR backup |
| F03 | Task B TCN | False ESCALATE (false alarm) | Unnecessary emergency | Medium | Low | 6 | 3-cycle hysteresis |
| F04 | GPS | IP geolocation error (3km) | Wrong hospital routed | High | Medium | 12 | Show distance, let driver confirm |
| F05 | OSM query | Network timeout | No hospital route | High | Low | 9 | Cached nearest ER fallback |
| F06 | LLM API | OpenAI timeout | No explanation | Low | Medium | 6 | Rule-based fallback |
| F07 | State machine | Stuck in ESCALATE | Continuous false alerts | High | Low | 9 | Manual reset button |
| F08 | Voice alert | macOS TTS failure | Silent alert | Medium | Low | 4 | Visual dashboard alert |
| F09 | WebSocket | Connection drop | Dashboard not updated | Low | Medium | 4 | Auto-reconnect |
| F10 | CUDA kernel | GPU OOM | Pipeline crash | High | Low | 9 | NumPy fallback |
| F11 | SQI | All channels low quality | Permanent abstain | High | Low | 9 | Timeout override after 30s |
| F12 | Emergency SMS | Twilio failure | Contact not notified | Critical | Low | 12 | macOS Messages fallback |

RPN = Severity x Likelihood x Detectability (1-3 scale each)

## Fail-Safe Behaviors

1. SQI abstain: hold previous state, never escalate on bad signal
2. GPU OOM: NumPy fallback, pipeline continues at reduced speed
3. LLM timeout: rule-based explanation fires immediately
4. OSM timeout: display "Find nearest hospital" text, no crash
5. 3-cycle hysteresis: prevents single-frame false escalation

## Not Implemented (Future Work)
- Hardware watchdog for pipeline stall detection
- Redundant sensor channels for critical path
- Formal hazard analysis per ISO 26262
- Clinical risk classification per IEC 62304

## Disclaimer
This FMEA is for research and educational purposes.
Guardian Drive is not a certified safety-critical system.
ISO 26262 / IEC 62304 compliance is not claimed.
