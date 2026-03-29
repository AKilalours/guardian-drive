"""List serial ports on the machine.

Run:
  python tools/list_serial_ports.py

macOS tip:
  Most Arduino-style devices show up under /dev/cu.usbmodem* or /dev/cu.usbserial*.
"""
from __future__ import annotations

try:
    from serial.tools import list_ports  # type: ignore
except Exception as e:
    raise SystemExit("pyserial not installed. Run: pip install -r requirements-integrations.txt")

ports = list(list_ports.comports())
if not ports:
    print("No serial ports found.")
else:
    for p in ports:
        print(f"{p.device}\t{p.description}")
