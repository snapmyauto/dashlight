# DashLight dashlight_v1_e100

- Date: 2026-10-27
- Base model: yolo11n.pt
- Epochs (max): 100, patience: 20, imgsz: 640, batch: 16
- Roboflow: snapmyauto/dashlight v1, train images: 3258
- Hardware: local GPU

## Test set

| mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|
| 0.847 | 0.613 | 0.852 | 0.798 |

## Per-class mAP@50-95

| Class | mAP@50-95 |
|---|---|
| abs | 0.772 |
| airbag | 0.739 |
| battery | 0.715 |
| brake_system | 0.630 |
| check_engine | 0.675 |
| coolant_temp | 0.738 |
| door_open | 0.453 |
| fog_light | 0.399 |
| low_fuel | 0.285 |
| oil_pressure | 0.690 |
| power_steering | 0.386 |
| seatbelt | 0.380 |
| traction_control | 0.742 |
| tyre_pressure | 0.799 |
| washer_fluid | 0.793 |
