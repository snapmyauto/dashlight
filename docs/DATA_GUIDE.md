# DashLight — Data Collection & Labelling Guide

## 1. Collect photos (everyone)

**Best trick:** turn the key to **ON (engine OFF)**. Most cars light up almost every warning light for a bulb check.

Take photos of:
- Different vehicles: cars, vans, SUVs, motorbikes, three-wheelers
- Different brands: Toyota, Suzuki, Honda, Nissan, Hyundai, Bajaj, etc.
- Different angles: straight, left, right, slightly above
- Different distances: close-up of one light, whole cluster
- Different lighting: daytime, evening, night, with glare
- Different phones (each member's phone)

Also take **real "problem" photos** when a light is on while driving (from a parked car only).

Rules:
- Don't photograph while driving.
- Keep number plates and faces out of the shot, or blur them.
- Target: **100+ images per class**. Weak classes need more.

Upload all photos to the shared folder / Roboflow project.

## 2. Roboflow setup (one person)

1. Sign up at roboflow.com → create Workspace **snapmyauto** → invite all members.
2. Create Project → **Object Detection** → name `dashlight`.
3. Add classes exactly as in `config/classes.txt`.

## 3. Add public datasets

1. Search Roboflow Universe for "dashboard warning light".
2. Check each dataset: classes, image quality, **licence**.
3. Use "Clone / Download" → add to the `dashlight` project.
4. Record name, link, image count and licence in the README dataset table.
5. Rename/merge class names so they match `classes.txt` (Roboflow: **Modify Classes**).
   Delete classes you don't use.

## 4. Label (split the work between members)

- Draw a **tight box** around each light icon — not the whole dashboard.
- Label **every** visible warning light in the image, not just one.
- Skip lights that are too blurry to recognise.
- Unsure which icon it is? Check the vehicle manual or ask the group. Don't guess.
- Use Roboflow **Label Assist** after the first version to speed up labelling.

## 5. Generate dataset version

Roboflow → Generate:
- Split: **70 / 20 / 10**
- Preprocessing: Auto-Orient, Resize **640×640 (fit/stretch)**
- Augmentation (×3):
  - Brightness ±25%
  - Blur up to 1.5px
  - Noise up to 2%
  - Rotation ±15°
  - **No horizontal flip**
- Generate → note version number → use it in the notebook.

## 6. Quality checklist before training

- [ ] All 20 classes have at least 50 images (aim for 100+)
- [ ] No class has 10× more images than another (balance)
- [ ] Test split includes group-collected phone photos
- [ ] Licences recorded
