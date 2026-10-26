# DashLight

On-device object detection model that recognises **car dashboard warning lights** from a phone camera image.
Built for the [SnapMyAuto](https://github.com/snapmyauto/snapmyauto-app) mobile app.

**Latest release:** [v1.0](https://github.com/snapmyauto/dashlight/releases/tag/v1.0) · test mAP@50 **0.847**

## Download

| File | Link |
|---|---|
| Model (TFLite / LiteRT, float32, 10.7 MB) | [dashlight.tflite](https://github.com/snapmyauto/dashlight/releases/download/v1.0/dashlight.tflite) |
| Class labels (order matches model output) | [labels.txt](https://github.com/snapmyauto/dashlight/releases/download/v1.0/labels.txt) |

## Model card

| Item | Value |
|---|---|
| Task | Object detection (bounding box + class) |
| Architecture | Ultralytics YOLO11n (transfer learning from COCO-pretrained weights) |
| Parameters | 2.6 M |
| Input | 640 × 640 RGB |
| Output | `(1, 19, 8400)`: 4 box values + 15 class scores per candidate |
| Format | TFLite / LiteRT (`.tflite`) for Android and iOS |
| Inference speed | ~5 ms per image (RTX 3050 Laptop GPU, PyTorch) |
| Training hardware | NVIDIA RTX 3050 Laptop GPU (4 GB), 1.5 h |

### Classes (v1.0)

`abs`, `airbag`, `battery`, `brake_system`, `check_engine`, `coolant_temp`, `door_open`, `fog_light`, `low_fuel`, `oil_pressure`, `power_steering`, `seatbelt`, `traction_control`, `tyre_pressure`, `washer_fluid`

Planned for later versions (not enough training data yet): `glow_plug`, `transmission_temp`, `dpf`, `high_beam`, `service_due`. The full planned list is in [`config/classes.txt`](config/classes.txt).

## Results

### v1.0: test set (465 unseen images)

| mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|
| **0.847** | 0.613 | 0.852 | 0.798 |

### Per class (test set)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| tyre_pressure | 0.933 | 0.910 | 0.959 | 0.799 |
| abs | 0.903 | 0.921 | 0.950 | 0.772 |
| airbag | 0.910 | 0.901 | 0.938 | 0.739 |
| traction_control | 0.895 | 0.858 | 0.937 | 0.742 |
| coolant_temp | 0.931 | 0.816 | 0.905 | 0.738 |
| washer_fluid | 0.785 | 0.900 | 0.905 | 0.793 |
| battery | 0.916 | 0.840 | 0.903 | 0.715 |
| oil_pressure | 0.888 | 0.827 | 0.891 | 0.690 |
| check_engine | 0.874 | 0.817 | 0.890 | 0.675 |
| brake_system | 0.882 | 0.792 | 0.882 | 0.630 |
| power_steering | 0.837 | 0.711 | 0.760 | 0.386 |
| door_open | 0.797 | 0.715 | 0.755 | 0.453 |
| seatbelt | 0.777 | 0.627 | 0.729 | 0.380 |
| low_fuel | 0.723 | 0.635 | 0.653 | 0.285 |
| fog_light | 0.725 | 0.692 | 0.643 | 0.399 |

Training charts, confusion matrix and the full metrics are in [`results/dashlight_v1_e100/`](results/dashlight_v1_e100/).

### Version history

| Version | Data | Epochs | mAP@50 | mAP@50-95 | Notes |
|---|---|---|---|---|---|
| test run | Roboflow v1 | 20 | 0.798 | 0.563 | Google Colab T4, pipeline check |
| **v1.0** | Roboflow v1 | 100 | **0.847** | **0.613** | Local RTX 3050, batch 16 |

## Dataset

Managed in Roboflow (workspace `snapmyauto`, project `dashlight`, version 1, public).

| | Images |
|---|---|
| Train | 3,258 |
| Validation | 931 |
| Test | 465 |
| **Total** | **4,654** |

Split 70 / 20 / 10. Preprocessing: auto-orient, resize (stretch) to 640 × 640. No offline augmentation; YOLO applies its own during training (mosaic, HSV, scale, translate). Horizontal flip is **disabled** because warning icons are not symmetric.

### Sources

| Source | Licence | How it was used |
|---|---|---|
| [AutoBud dashboard-ai](https://universe.roboflow.com/autobud-workspace/dashboard-ai) | CC BY 4.0 | Imported, classes remapped |
| [Car Dashboard Detection](https://universe.roboflow.com/car-dashboard-detection/car-dashboard-detection) | MIT | Imported, classes remapped |
| [Dashboard icons](https://universe.roboflow.com/shourya-chambial-axqy0/dashboard-icons) (Shourya Chambial) | CC BY 4.0 | Imported, classes remapped |
| [Tell-Tale Detection - 2](https://universe.roboflow.com/pm-sxjpn/tell-tale-detection-2) (PM) | CC BY 4.0 | Augmented copies removed (7,456 → 4,553 unique images), 38 classes remapped/dropped to match this project |

Cleaning steps:
- **De-duplication:** the Tell-Tale export contained 3–5 augmented copies of each photo, and 504 photos appeared in both train and test. Only one copy of each photo was kept, then everything was re-split. This prevents test-set leakage.
- **Class remapping:** names from all sources were merged into one naming scheme (e.g. `Braking System Issue`, `Brake Warning Light` → `brake_system`).
- **Dropped classes:** master warning, lane departure, low beam, side lamp, security, automatic shift lock (not in scope); `transmission_temp` (only 2 examples).

## How to train

### Option A: local GPU (recommended)

Requires an NVIDIA GPU (4 GB+), Python 3.10+.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash  (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt

read -s -p "Roboflow key: " ROBOFLOW_API_KEY && export ROBOFLOW_API_KEY && echo
python scripts/train_local.py --batch 16             # full run (100 epochs, early stopping)
python scripts/train_local.py --epochs 1 --name smoke_test   # quick pipeline check
python scripts/train_local.py --resume --name dashlight_v1_e100   # resume after interruption
```

The dataset, weights and runs are stored in `~/dashlight_work` (outside the repo). Charts and `metrics.md` are copied into `results/<run name>/`.

### Option B: Google Colab

Open [`notebooks/train.ipynb`](notebooks/train.ipynb) in Colab → T4 GPU → add `ROBOFLOW_API_KEY` in Colab Secrets → Run all.
The free tier may disconnect on long runs; see the resume cell.

### Export to TFLite

LiteRT export is only supported on **Linux x86 / macOS** (not Windows). On Windows, export in a Colab **CPU** runtime:

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="litert", imgsz=640)   # -> best.tflite
```

## Using the model in the app

1. Download `dashlight.tflite` and `labels.txt` from the release.
2. Place them in `snapmyauto-app/assets/models/`.
3. Input: 640 × 640 RGB, normalised 0–1. Output: `(1, 19, 8400)`. Apply a confidence threshold (e.g. 0.5) and non-max suppression.

## Limitations

- Recognises the 15 classes above only. Other icons may be missed or matched to the closest class, so the app should fall back to manual lookup or the Gemini fallback when confidence is low.
- Lower accuracy on `fog_light`, `low_fuel` and `seatbelt` (fewer training examples).
- Most training images come from public datasets; accuracy may drop with heavy glare, blur, night shots or vehicle brands not represented in the data.
- Light **colour** (red / amber / green) and **blinking** are detected by the app, not by this model.
- Decision support only, not a replacement for a qualified mechanic.

## Roadmap

- Add group-collected photos from Sri Lankan vehicles (cars, vans, motorbikes, three-wheelers).
- Collect data for the missing classes (`glow_plug`, `transmission_temp`, `dpf`, `high_beam`, `service_due`).
- INT8 quantised model for faster inference on low-end phones.

## Licence

The model was trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under **AGPL-3.0**; this repository and model are released under the same licence (see [LICENSE](LICENSE)).
Dataset licences are listed in the Sources table above.