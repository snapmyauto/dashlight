# DashLight

On-device object detection model that recognises **car dashboard warning lights** from a phone camera image.
Built for the [SnapMyAuto](https://github.com/snapmyauto/snapmyauto-app) mobile app.

## Model card

| Item | Value |
|---|---|
| Task | Object detection (bounding box + class) |
| Architecture | Ultralytics YOLO11n (transfer learning from COCO weights) |
| Input size | 640 × 640 RGB |
| Output format | TensorFlow Lite (`.tflite`) for Android / iOS |
| Classes | 20 (see [`config/classes.txt`](config/classes.txt)) |
| Training platform | Google Colab (NVIDIA T4 GPU) |
| Dataset tool | Roboflow |

### Classes
`check_engine`, `oil_pressure`, `battery`, `coolant_temp`, `brake_system`, `abs`, `airbag`, `tyre_pressure`, `seatbelt`, `traction_control`, `glow_plug`, `low_fuel`, `door_open`, `power_steering`, `transmission_temp`, `dpf`, `high_beam`, `fog_light`, `washer_fluid`, `service_due`

### Dataset
| Source | Images | Licence |
|---|---|---|
| Group-collected photos (Sri Lanka) | _TBD_ | Own |
| Roboflow Universe dataset 1 | _TBD_ | _check_ |
| Roboflow Universe dataset 2 | _TBD_ | _check_ |

Split: 70% train / 20% validation / 10% test. Augmentation: brightness ±25%, blur, noise, rotation ±15° (no horizontal flip).

### Results
| Version | mAP@50 | mAP@50-95 | Precision | Recall | Date |
|---|---|---|---|---|---|
| v1.0 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Charts: [`results/`](results/)

### Limitations
- Trained on 20 common warning lights only; other icons are not recognised.
- Accuracy drops with heavy glare, motion blur or very dark images.
- Light colour (red/amber/green) and blinking are detected by the app, not by this model.
- Not a replacement for a qualified mechanic.

## How to train
1. Prepare and label the dataset in Roboflow — see [`docs/DATA_GUIDE.md`](docs/DATA_GUIDE.md).
2. Open [`notebooks/train.ipynb`](notebooks/train.ipynb) in Google Colab.
3. Add your Roboflow API key in Colab **Secrets** as `ROBOFLOW_API_KEY`.
4. Runtime → Change runtime type → **T4 GPU** → Run all.
5. Download `dashlight_export.zip` (model + labels + metrics).

## How to use in the app
Copy `dashlight.tflite` and `labels.txt` into `snapmyauto-app/assets/models/`.

## Licence
Trained with Ultralytics YOLO, which is licensed under **AGPL-3.0**; this model is released under the same licence.
Dataset licences are listed above.
