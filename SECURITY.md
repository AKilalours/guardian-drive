# Guardian Drive -- Cybersecurity Threat Model

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis

## Threat Model (STRIDE)

### S -- Spoofing
- Threat: Attacker spoofs sensor data (fake ECG, fake GPS)
- Impact: False NOMINAL state, missed emergency
- Mitigation: Signal quality index (SQI) detects implausible signals
- Mitigation: Cryptographic signing of sensor frames (future work)

### T -- Tampering
- Threat: Attacker modifies telemetry JSONL logs
- Impact: Corrupted audit trail, false regression results
- Mitigation: HMAC per record in TelemetryLogger (future work)
- Mitigation: Append-only log with hash chaining

### R -- Repudiation
- Threat: No audit trail of safety decisions
- Impact: Cannot investigate post-incident
- Mitigation: TelemetryLogger writes every frame to JSONL
- Mitigation: TelemetryReplayer enables forensic replay

### I -- Information Disclosure
- Threat: Physiological data exposed via WebSocket
- Impact: Privacy violation (ECG, location, health data)
- Mitigation: WSS (WebSocket Secure) in production
- Mitigation: Local-only by default (127.0.0.1)
- Mitigation: No cloud upload of raw sensor data

### D -- Denial of Service
- Threat: Attacker floods WebSocket, delays emergency response
- Impact: State machine stalls, alert not fired
- Mitigation: Rate limiting on WebSocket connections
- Mitigation: Watchdog timer fires alert if pipeline stalls >5s

### E -- Elevation of Privilege
- Threat: Compromised LLM response triggers false ESCALATE
- Impact: Unnecessary emergency dispatch
- Mitigation: LLM output is advisory only, never triggers actions directly
- Mitigation: State machine requires 3 consecutive cycles to escalate

## API Key Security
- OPENAI_API_KEY: stored in .env, never committed
- Discord webhook: rotated immediately on exposure
- Twilio credentials: .env only

## Data Privacy
- ECG, EDA signals: processed locally, never sent to cloud
- GPS location: sent to OSM only on ESCALATE (hospital routing)
- Emergency SMS: sent only on ESCALATE with explicit contact config

## Not Yet Implemented
- mTLS for WebSocket in production
- Hardware security module (HSM) for key storage
- Differential privacy for fleet telemetry aggregation
- Formal verification of state machine transitions
