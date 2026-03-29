import os, shutil, random, yaml, tarfile, json
from pathlib import Path
import cv2
import numpy as np
import albumentations as A

def load_yaml_config(filename: str):
    project_root = Path.cwd()  # works in Colab + local
    config_path = project_root / "configs" / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

assert (Path.cwd() / "configs").exists(), "configs/ folder missing in project root"

cfg = load_yaml_config("yolo.yaml")["yolo"]

SEED = cfg["seed"]
random.seed(SEED)
np.random.seed(SEED)

YOLO_MODEL = cfg["model"]
DATASET_NAME = cfg["dataset"]["name"]

IMG_SIZE = cfg["training"]["img_size"]
EPOCHS = cfg["training"]["epochs"]
PATIENCE = cfg["training"]["patience"]
BATCH = cfg["training"]["batch"]
DEVICE = cfg.get("device", 0)

TARGET_TRAIN_EGG_IMAGES = cfg["augmentation"]["targets"]["train_egg_images"]
TARGET_TRAIN_LARVAE_IMAGES = cfg["augmentation"]["targets"]["train_larvae_images"]

TARGET_VAL_EGG_IMAGES = cfg["augmentation"]["targets"]["val_egg_images"]
VAL_NON_EGG = cfg["augmentation"]["targets"]["val_non_egg"]

MAX_AUG_PER_SOURCE = cfg["augmentation"]["max_aug_per_source"]

CLASS_NAMES = cfg["classes"]["names"]
LARVAE_CLS = cfg["classes"]["larvae_id"]
EGG_CLS = cfg["classes"]["egg_id"]

SRC_ROOT = Path(cfg["dataset"]["root"])

OUT_ROOT = Path(cfg["output"]["trainval_dir"])
OUT_IMG_TRAIN = OUT_ROOT / "images/train"
OUT_IMG_VAL   = OUT_ROOT / "images/val"
OUT_IMG_TEST  = OUT_ROOT / "images/test"
OUT_LBL_TRAIN = OUT_ROOT / "labels/train"
OUT_LBL_VAL   = OUT_ROOT / "labels/val"
OUT_LBL_TEST  = OUT_ROOT / "labels/test"

RUNS_DIR = Path(cfg["output"]["runs_dir"])
OP_NAME  = f"augEggLarvaeTrainVal_{IMG_SIZE}_e{EPOCHS}"
RUN_NAME = f"{YOLO_MODEL}_{DATASET_NAME}_exp"

print("SRC_ROOT:", SRC_ROOT)
print("OUT_ROOT:", OUT_ROOT)
print("RUNS_DIR:", RUNS_DIR)
print("RUN_NAME:", RUN_NAME)

print("\n===== CONFIG SUMMARY =====")
print("Seed:", SEED)
print("Model:", YOLO_MODEL)
print("Dataset:", DATASET_NAME)
print("Image size:", IMG_SIZE)
print("Epochs:", EPOCHS)
print("Batch:", BATCH)
print("Device:", DEVICE)
print("==========================\n")
# ----------------------------
# Helpers
# ----------------------------
def read_yolo_labels(lbl_path: Path):
    if not lbl_path.exists():
        return []
    rows = []
    with open(lbl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) != 5:
                continue
            cls, x, y, w, h = p
            rows.append([int(cls), float(x), float(y), float(w), float(h)])
    return rows

def write_yolo_labels(lbl_path: Path, rows):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w") as f:
        for cls, x, y, w, h in rows:
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def list_pairs(split: str):
    img_dir = SRC_ROOT / f"images/{split}"
    lbl_dir = SRC_ROOT / f"labels/{split}"
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    imgs = []
    for ext in exts:
        imgs += list(img_dir.glob(f"*{ext}"))
    pairs = []
    for img in imgs:
        pairs.append((img, lbl_dir / (img.stem + ".txt")))
    return pairs

def scan_split(img_dir, lbl_dir):
    img_paths = []
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        img_paths += list(Path(img_dir).glob(f"*{ext}"))
    img_paths = sorted(img_paths)

    imgs_per_class = np.zeros(len(CLASS_NAMES), dtype=int)
    inst_per_class = np.zeros(len(CLASS_NAMES), dtype=int)

    for imgp in img_paths:
        lblp = Path(lbl_dir) / (imgp.stem + ".txt")
        rows = read_yolo_labels(lblp)
        present = set()
        for r in rows:
            c = r[0]
            if 0 <= c < len(CLASS_NAMES):
                inst_per_class[c] += 1
                present.add(c)
        for c in present:
            imgs_per_class[c] += 1

    return len(img_paths), imgs_per_class, inst_per_class

def print_scan(title, img_dir, lbl_dir):
    total, imgs_c, inst_c = scan_split(img_dir, lbl_dir)
    print(f"\n{title}  Total images: {total}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:15s} imgs={imgs_c[i]:4d}  inst={inst_c[i]:5d}")

def copy_pairs(pairs, out_img_dir, out_lbl_dir):
    out_img_dir = Path(out_img_dir); out_lbl_dir = Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    for imgp, lblp in pairs:
        shutil.copy2(imgp, out_img_dir / imgp.name)
        dst_lbl = out_lbl_dir / (imgp.stem + ".txt")
        if lblp.exists():
            shutil.copy2(lblp, dst_lbl)
        else:
            write_yolo_labels(dst_lbl, [])

def has_class(lbl_path: Path, cls_id: int):
    rows = read_yolo_labels(lbl_path)
    return any(r[0] == cls_id for r in rows)

def current_class_image_count(img_dir, lbl_dir, cls_id: int):
    img_paths = []
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        img_paths += list(Path(img_dir).glob(f"*{ext}"))
    c = 0
    for imgp in img_paths:
        lblp = Path(lbl_dir) / (imgp.stem + ".txt")
        if has_class(lblp, cls_id):
            c += 1
    return c

def count_bad_images(img_dir):
    bad = 0
    img_paths = []
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        img_paths += list(Path(img_dir).glob(f"*{ext}"))
    for imgp in img_paths:
        im = cv2.imread(str(imgp))
        if im is None or im.size == 0:
            bad += 1
            continue
        h, w = im.shape[:2]
        if h < 2 or w < 2:
            bad += 1
    return bad

def yolo_to_xyxy(bb, W, H):
    x, y, w, h = bb
    xc = x * W; yc = y * H
    bw = w * W; bh = h * H
    x1 = int(max(0, xc - bw/2)); y1 = int(max(0, yc - bh/2))
    x2 = int(min(W-1, xc + bw/2)); y2 = int(min(H-1, yc + bh/2))
    return x1, y1, x2, y2

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw/2
    yc = y1 + bh/2
    return float(xc / W), float(yc / H), float(bw / W), float(bh / H)

# ----------------------------
# 2) Prepare output dirs (clean)
# ----------------------------
for p in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_IMG_TEST, OUT_LBL_TRAIN, OUT_LBL_VAL, OUT_LBL_TEST]:
    p.mkdir(parents=True, exist_ok=True)

for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_IMG_TEST, OUT_LBL_TRAIN, OUT_LBL_VAL, OUT_LBL_TEST]:
    for f in d.glob("*"):
        f.unlink()

# ----------------------------
# 3) Resplit (NO test leakage)
# ----------------------------
all_pairs = []
for split in ["train", "val"]:
    all_pairs += list_pairs(split)
test_pairs = list_pairs("test")  # untouched

egg_pairs, non_egg_pairs = [], []
for imgp, lblp in all_pairs:
    rows = read_yolo_labels(lblp)
    has_egg = any(r[0] == EGG_CLS for r in rows)
    (egg_pairs if has_egg else non_egg_pairs).append((imgp, lblp))

random.shuffle(egg_pairs)
random.shuffle(non_egg_pairs)

val_egg   = egg_pairs[:TARGET_VAL_EGG_IMAGES]
train_egg = egg_pairs[TARGET_VAL_EGG_IMAGES:]

val_non_egg   = non_egg_pairs[:VAL_NON_EGG]
train_non_egg = non_egg_pairs[VAL_NON_EGG:]

val_pairs   = val_egg + val_non_egg
train_pairs = train_egg + train_non_egg
random.shuffle(val_pairs); random.shuffle(train_pairs)

print("\nResplit summary (train+val only, NO test leakage):")
print(" - total (train+val):", len(all_pairs))
print(" - egg images total:", len(egg_pairs))
print(" - val egg:", len(val_egg))
print(" - train egg:", len(train_egg))
print(" - val total:", len(val_pairs))
print(" - train total:", len(train_pairs))
print(" - original test kept aside:", len(test_pairs))

copy_pairs(train_pairs, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
copy_pairs(val_pairs,   OUT_IMG_VAL,   OUT_LBL_VAL)
copy_pairs(test_pairs,  OUT_IMG_TEST,  OUT_LBL_TEST)

print_scan("TRAIN (after resplit, before aug)", OUT_IMG_TRAIN, OUT_LBL_TRAIN)
print_scan("VAL   (after resplit, before aug)", OUT_IMG_VAL,   OUT_LBL_VAL)
print_scan("TEST  (untouched)",                OUT_IMG_TEST,  OUT_LBL_TEST)

# ----------------------------
# 4) Oversampling augmentation
#   Eggs: bbox-safe general augmentation
#   Larvae: COPY-PASTE (strong) + bbox-safe top-up if needed
# ----------------------------
aug_general = A.Compose(
    [
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.65, 1.0), ratio=(0.85, 1.15), p=0.9),
        A.Affine(rotate=(-15, 15), scale=(0.80, 1.20), translate_percent=(-0.08, 0.08), p=0.85),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.30),
        A.MotionBlur(blur_limit=7, p=0.25),
        A.CLAHE(p=0.25),
        A.Sharpen(p=0.15),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.15),
)

def oversample_with_general_aug(cls_id: int, target_images: int, tag: str):
    now = current_class_image_count(OUT_IMG_TRAIN, OUT_LBL_TRAIN, cls_id)
    need = max(0, target_images - now)
    print(f"\n[{tag}] general-aug class {cls_id}: now={now} target={target_images} need={need}")
    if need == 0:
        return 0

    train_imgs = []
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        train_imgs += list(OUT_IMG_TRAIN.glob(f"*{ext}"))
    train_imgs = sorted(train_imgs)

    sources = []
    for imgp in train_imgs:
        lblp = OUT_LBL_TRAIN / (imgp.stem + ".txt")
        if has_class(lblp, cls_id):
            sources.append((imgp, lblp))

    if not sources:
        print(f"[{tag}] WARNING: no sources for cls={cls_id}")
        return 0

    gen, src_idx = 0, 0
    while gen < need:
        imgp, lblp = sources[src_idx % len(sources)]
        src_idx += 1
        im = cv2.imread(str(imgp))
        if im is None:
            continue

        rows = read_yolo_labels(lblp)
        bboxes = [[r[1], r[2], r[3], r[4]] for r in rows]
        clabels = [r[0] for r in rows]

        for _ in range(MAX_AUG_PER_SOURCE):
            if gen >= need:
                break
            try:
                out = aug_general(image=im, bboxes=bboxes, class_labels=clabels)
            except Exception:
                continue
            if cls_id not in set(out["class_labels"]):
                continue

            new_name = f"{imgp.stem}_{tag}_{gen:05d}.jpg"
            new_img_path = OUT_IMG_TRAIN / new_name
            new_lbl_path = OUT_LBL_TRAIN / (Path(new_name).stem + ".txt")

            cv2.imwrite(str(new_img_path), out["image"])
            new_rows = []
            for bb, c in zip(out["bboxes"], out["class_labels"]):
                x, y, w, h = bb
                x = float(np.clip(x, 0, 1)); y = float(np.clip(y, 0, 1))
                w = float(np.clip(w, 0, 1)); h = float(np.clip(h, 0, 1))
                if w <= 0 or h <= 0:
                    continue
                new_rows.append([int(c), x, y, w, h])
            write_yolo_labels(new_lbl_path, new_rows)
            gen += 1

    print(f"[{tag}] generated={gen}")
    return gen

def build_crop_bank(cls_id: int, max_crops: int = 1500):
    bank = []
    train_imgs = []
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        train_imgs += list(OUT_IMG_TRAIN.glob(f"*{ext}"))
    random.shuffle(train_imgs)

    for imgp in train_imgs:
        lblp = OUT_LBL_TRAIN / (imgp.stem + ".txt")
        rows = read_yolo_labels(lblp)
        targets = [r for r in rows if r[0] == cls_id]
        if not targets:
            continue

        im = cv2.imread(str(imgp))
        if im is None:
            continue
        H, W = im.shape[:2]

        for r in targets:
            _, x, y, w, h = r
            x1, y1, x2, y2 = yolo_to_xyxy([x, y, w, h], W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = im[y1:y2, x1:x2].copy()
            ch, cw = crop.shape[:2]
            if ch < 6 or cw < 6:
                continue
            bank.append(crop)
            if len(bank) >= max_crops:
                return bank
    return bank

def paste_crop_on_image(base_img, crop, alpha_blend=True):
    img = base_img.copy()
    H, W = img.shape[:2]
    ch, cw = crop.shape[:2]

    scale = random.uniform(0.6, 1.4)
    nw = max(4, int(cw * scale))
    nh = max(4, int(ch * scale))
    crop_r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

    nh, nw = crop_r.shape[:2]
    if nh >= H or nw >= W:
        return None, None

    x1 = random.randint(0, W - nw - 1)
    y1 = random.randint(0, H - nh - 1)
    x2 = x1 + nw
    y2 = y1 + nh

    if alpha_blend:
        mask = np.ones((nh, nw), dtype=np.float32)
        k = max(3, int(min(nh, nw) * 0.08))
        k = k + 1 if k % 2 == 0 else k
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = mask[..., None]
        patch = crop_r.astype(np.float32)
        roi = img[y1:y2, x1:x2].astype(np.float32)
        out = roi * (1.0 - mask) + patch * mask
        img[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)
    else:
        img[y1:y2, x1:x2] = crop_r

    return img, (x1, y1, x2, y2)

def oversample_with_copypaste(cls_id: int, target_images: int, tag: str):
    now = current_class_image_count(OUT_IMG_TRAIN, OUT_LBL_TRAIN, cls_id)
    need = max(0, target_images - now)
    print(f"\n[{tag}] copy-paste class {cls_id}: now={now} target={target_images} need={need}")
    if need == 0:
        return 0

    bank = build_crop_bank(cls_id, max_crops=1500)
    print(f"[{tag}] crop bank size:", len(bank))
    if len(bank) == 0:
        print(f"[{tag}] WARNING: empty crop bank for cls={cls_id}")
        return 0

    bg_imgs = []
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        bg_imgs += list(OUT_IMG_TRAIN.glob(f"*{ext}"))
    random.shuffle(bg_imgs)
    if len(bg_imgs) == 0:
        print(f"[{tag}] WARNING: no train images to paste into")
        return 0

    gen = 0
    tries = 0
    while gen < need and tries < need * 40:
        tries += 1
        bgp = bg_imgs[tries % len(bg_imgs)]
        base = cv2.imread(str(bgp))
        if base is None:
            continue
        H, W = base.shape[:2]

        lblp = OUT_LBL_TRAIN / (bgp.stem + ".txt")
        rows = read_yolo_labels(lblp)

        crop = bank[random.randint(0, len(bank)-1)]
        pasted, xyxy = paste_crop_on_image(base, crop, alpha_blend=True)
        if pasted is None:
            continue

        x1, y1, x2, y2 = xyxy
        x, y, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
        if w <= 0 or h <= 0:
            continue

        new_name = f"{bgp.stem}_{tag}_{gen:05d}.jpg"
        new_img_path = OUT_IMG_TRAIN / new_name
        new_lbl_path = OUT_LBL_TRAIN / (Path(new_name).stem + ".txt")

        new_rows = rows + [[cls_id, x, y, w, h]]
        cv2.imwrite(str(new_img_path), pasted)
        write_yolo_labels(new_lbl_path, new_rows)
        gen += 1

    print(f"[{tag}] generated={gen} (tries={tries})")
    return gen

# Eggs (general aug)
gen_egg = oversample_with_general_aug(EGG_CLS, TARGET_TRAIN_EGG_IMAGES, tag="augEgg")
# Larvae (copy-paste strong) + top-up
gen_larvae_cp  = oversample_with_copypaste(LARVAE_CLS, TARGET_TRAIN_LARVAE_IMAGES, tag="cpLarvae")
gen_larvae_top = oversample_with_general_aug(LARVAE_CLS, TARGET_TRAIN_LARVAE_IMAGES, tag="augLarvaeTopUp")

print_scan("TRAIN (after egg+larvae oversampling)", OUT_IMG_TRAIN, OUT_LBL_TRAIN)
print_scan("VAL   (unchanged)",                   OUT_IMG_VAL,   OUT_LBL_VAL)
print_scan("TEST  (unchanged)",                   OUT_IMG_TEST,  OUT_LBL_TEST)

print("\nBad train:", count_bad_images(OUT_IMG_TRAIN))
print("Bad val:",   count_bad_images(OUT_IMG_VAL))
print("Bad test:",  count_bad_images(OUT_IMG_TEST))

# ----------------------------
# 5) Write dataset YAML
# ----------------------------
yaml_path = Path("/content/data_aug_trainval.yaml")
data_yaml = {
    "path": str(OUT_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
with open(yaml_path, "w") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)

print("\nWrote:", yaml_path)
print(open(yaml_path).read())

# ----------------------------
# 6) Train YOLOv8m
#    IMPORTANT: use SAFE-ARGS filter so unsupported args never crash training
# ----------------------------
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT  # contains supported keys for this version

RUNS_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(f"{YOLO_MODEL}.pt")

# Train args you want (some may vary by version, so we filter)
train_kwargs = dict(
    data=str(yaml_path),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    patience=PATIENCE,
    name=RUN_NAME,
    project=str(RUNS_DIR),
    exist_ok=True,

    # stability
    multi_scale=False,
    rect=False,

    # saving
    save=True,
    save_period=1,

    # schedule + stronger built-in aug (if supported)
    cos_lr=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.10,

    # practical
    workers=4,
)

# SAFE FILTER
allowed = set(DEFAULT_CFG_DICT.keys())
train_kwargs_filtered = {k: v for k, v in train_kwargs.items() if k in allowed}

dropped = sorted(set(train_kwargs.keys()) - set(train_kwargs_filtered.keys()))
print("\nTraining args (filtered):\n", json.dumps(train_kwargs_filtered, indent=2))
if dropped:
    print("\n⚠️ Dropped unsupported args for this Ultralytics version:", dropped)

try:
    model.train(**train_kwargs_filtered)
except Exception as e:
    print(f"\n❌ YOLO training failed: {e}")
    raise

run_dir   = RUNS_DIR / RUN_NAME
best_path = run_dir / "weights/best.pt"
last_path = run_dir / "weights/last.pt"

if run_dir.exists():
    with open(run_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

print("\n✅ Training finished.")
print("Run folder:", run_dir)
print("Weights:")
print(" - best:", best_path, "exists:", best_path.exists())
print(" - last:", last_path, "exists:", last_path.exists())

# ----------------------------
# 7) Evaluation (VAL, TEST) + optional TTA
# ----------------------------
if best_path.exists():
    best_model = YOLO(str(best_path))

    print("\n📌 VAL metrics (standard):")
    best_model.val(data=str(yaml_path), imgsz=IMG_SIZE, split="val")

    print("\n📌 TEST metrics (standard, untouched):")
    best_model.val(data=str(yaml_path), imgsz=IMG_SIZE, split="test")

    print("\n📌 VAL metrics with TTA (augment=True):")
    best_model.val(data=str(yaml_path), imgsz=IMG_SIZE, split="val", augment=True)

    print("\n📌 TEST metrics with TTA (augment=True):")
    best_model.val(data=str(yaml_path), imgsz=IMG_SIZE, split="test", augment=True)
else:
    print("\n⚠️ best.pt not found. Check:", run_dir / "weights")

# ----------------------------
# 8) Professional plot names (COPY originals with clearer names)
# ----------------------------
rename_map = {
    "results.png": "training_curves_losses_precision_recall_map.png",
    "results.csv": "training_log_metrics_per_epoch.csv",

    "BoxP_curve.png":  "precision_vs_confidence_curve.png",
    "BoxR_curve.png":  "recall_vs_confidence_curve.png",
    "BoxF1_curve.png": "f1_score_vs_confidence_curve.png",
    "BoxPR_curve.png": "precision_recall_curve.png",

    "confusion_matrix.png": "confusion_matrix_counts.png",
    "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",

    "labels.jpg": "dataset_label_distribution_and_bbox_statistics.jpg",
    "labels.png": "dataset_label_distribution_and_bbox_statistics.png",
}

def safe_copy(src: Path, dst: Path):
    try:
        if src.exists():
            shutil.copy2(src, dst)
            return True
    except Exception:
        pass
    return False

if run_dir.exists():
    assets_dir = run_dir / "professional_named_plots"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_name, dst_name in rename_map.items():
        if safe_copy(run_dir / src_name, assets_dir / dst_name):
            copied += 1
    print(f"\n🧾 Professional plot copies saved in: {assets_dir}")
    print(f"Copied {copied}/{len(rename_map)} files.")
else:
    print("\n⚠️ Run dir not found, cannot rename plots:", run_dir)

# ----------------------------
# 9) Archive the entire run folder to Drive (.tar.gz)
# ----------------------------
tar_path = RUNS_DIR / f"{RUN_NAME}.tar.gz"
if run_dir.exists():
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    print("\n✅ Saved archive:", tar_path)
else:
    print("\n⚠️ Run dir not found, cannot archive:", run_dir)

print("\n✅ DONE.")
print("Key outputs:")
print(" - Run folder:", run_dir)
print(" - Best weights:", best_path)
print(" - Archive:", tar_path)
print(" - Professional plots folder:", run_dir / "professional_named_plots")
