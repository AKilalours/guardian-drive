# Guardian Drive -- Safety & Disclaimer

## NOT A MEDICAL DEVICE

Guardian Drive is a research and portfolio project.

It has NOT been:
- Clinically validated
- Approved by FDA or any regulatory body
- Tested in real vehicles
- Verified for medical-grade accuracy

## Claim Guardrail

The system explicitly sets medical_grade=False in policy/fusion.py.
The system abstains from inference when signal quality is insufficient.
All model outputs are advisory only.

## Honest Limitations

| Limitation | Detail |
|-----------|--------|
| EAR threshold | Calibrated per face -- may need recalibration |
| Task B AUC | 0.9514 on WESAD -- not validated on general population |
| GPS accuracy | IP geolocation -- 3km radius, not precise |
| Hospital routing | OSM data -- may not reflect current hours or availability |
| Crash detection | IMU threshold -- not tested in real crash scenarios |

## Built By

Akila Lourdes Miriyala Francis & Akilan Manivannan
For research and portfolio demonstration purposes only.
