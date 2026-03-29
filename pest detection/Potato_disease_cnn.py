# ============================================================
# EfficientNet-B3 + MixUp + Cosine Scheduler (Kaggle-ready)
# - Dataset: ImageFolder with class subfolders
# - Splits: train/val/test (stratified)
# - Imbalance: WeightedRandomSampler (train) + class-weighted loss (optional)
# - Augment: RandAugment + RandomResizedCrop + ColorJitter + RandomErasing
# - MixUp: enabled during training
# - Scheduler: CosineAnnealingLR
# - Saves: history.csv + curves + confusion matrices + reports + .pt + zip
# ============================================================

import os, random, math, time, zipfile
from pathlib import Path
from dataclasses import dataclass
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = cfg.get("seed", 42)
seed_everything(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ----------------------------
# CONFIG
# ----------------------------
def load_yaml_config(filename: str):
    project_root = Path.cwd()
    config_path = project_root / "configs" / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

assert (Path.cwd() / "configs").exists(), "configs/ folder missing in project root"

cfg = load_yaml_config("cnn.yaml")["cnn"]
KAGGLE_INPUT_ROOT = cfg.get("kaggle_input_root", "/kaggle/input")

    # Training
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    IMG_SIZE: int = 300            # EfficientNet-B3 default is often 300
    LR: float = 3e-4
    WD: float = 1e-4
    NUM_WORKERS: int = 2

    # Split
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15         # from remaining after test split
    STRATIFY: bool = True

    # MixUp
    MIXUP_ALPHA: float = 0.4       # 0.2–0.4 typical
    MIXUP_PROB: float = 1.0        # apply mixup this fraction of batches

    # Loss
    USE_CLASS_WEIGHTS: bool = False  # start False; enable if minority suffers

    # Scheduler
    COSINE_TMAX: int = 50          # cosine period; can be EPOCHS as well
    MIN_LR: float = 1e-6

    # Early stopping
    EARLY_STOP_PATIENCE: int = 15

    # Output
    OUT_DIR: str = "/kaggle/working/leaf_cnn_runs"

cfg = CFG()

# ----------------------------
# Find DATA_ROOT containing class folders (ImageFolder)
# ----------------------------
def list_dir(p: Path):
    return [x for x in p.iterdir()]

def looks_like_class_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    subdirs = [p for p in d.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return False
    # must contain at least some images in subfolders
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    for sd in subdirs[:10]:
        if any(str(fp).lower().endswith(exts) for fp in sd.rglob("*")):
            return True
    return False

def find_class_root(root: Path) -> Path:
    # 1) direct root
    if looks_like_class_dir(root):
        return root

    # 2) search depth 4 for the first matching directory
    candidates = []
    for d in root.rglob("*"):
        if d.is_dir():
            # limit depth
            if len(d.relative_to(root).parts) > 6:
                continue
            if looks_like_class_dir(d):
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"Could not find ImageFolder class root under: {root}")
    # pick the shallowest
    candidates.sort(key=lambda p: len(p.relative_to(root).parts))
    return candidates[0]

kaggle_input = Path(KAGGLE_INPUT_ROOT)
print("\nAvailable /kaggle/input datasets:")
for p in kaggle_input.iterdir():
    if p.is_dir():
        print(" -", p.name)

# If you already know your dataset path, set it here directly:
# DATA_ROOT = Path("/kaggle/input/datasets/warcoder/potato-leaf-disease-dataset/Potato Leaf Disease Dataset in Uncontrolled Environment")
# Otherwise auto-find:
dataset_root = Path(cfg["dataset"]["root"])

if dataset_root.exists():
    DATA_ROOT = dataset_root
else:
    DATA_ROOT = find_class_root(kaggle_input)

print("\n✅ Using DATA_ROOT:", DATA_ROOT)
print("\n===== CONFIG SUMMARY =====")
print("Seed:", SEED)
print("Model:", cfg["model"])
print("Dataset:", cfg["dataset"]["name"])
print("Dataset source:", cfg["dataset"]["source"])
print("Image size:", cfg["image_size"])
print("Epochs:", cfg["epochs"])
print("Batch size:", cfg["batch_size"])
print("Learning rate:", cfg["learning_rate"])
print("==========================\n")

# ----------------------------
# Transforms
# ----------------------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(cfg["image_size"], scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
])

val_tf = transforms.Compose([
    transforms.Resize(int(cfg["image_size"] * 1.15)),
    transforms.CenterCrop(cfg["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# Load full dataset (base with val_tf, we override per split later)
# ----------------------------
base_ds = datasets.ImageFolder(DATA_ROOT, transform=val_tf)
class_names = base_ds.classes
num_classes = len(class_names)
print("\nClasses:", class_names, "num_classes:", num_classes)

# Targets for stratification
targets = np.array([base_ds.samples[i][1] for i in range(len(base_ds))])

# ----------------------------
# Split indices: train/val/test (stratified)
# ----------------------------
idx_all = np.arange(len(base_ds))

strat = targets if cfg["split"]["stratify"] else None
idx_trainval, idx_test = train_test_split(
    idx_all, test_size=cfg["split"]["test_size"], random_state=SEED, stratify=strat
)

targets_trainval = targets[idx_trainval]
strat2 = targets_trainval if cfg["split"]["stratify"] else None
val_fraction_of_trainval = cfg["split"]["val_size"] / (1.0 - cfg["split"]["test_size"])

idx_train, idx_val = train_test_split(
    idx_trainval, test_size=val_fraction_of_trainval, random_state=SEED, stratify=strat2
)

print("\nSplit sizes:", len(idx_train), len(idx_val), len(idx_test))

# Build subsets
train_ds = Subset(base_ds, idx_train)
val_ds   = Subset(base_ds, idx_val)
test_ds  = Subset(base_ds, idx_test)

# Override transform per subset (torchvision ImageFolder stores transform in dataset, but Subset wraps it)
# We can monkey-patch via accessing .dataset.transform, but that affects all subsets.
# Safer: create separate ImageFolder datasets and subset indices on each.

full_train_ds = datasets.ImageFolder(DATA_ROOT, transform=train_tf)
full_val_ds   = datasets.ImageFolder(DATA_ROOT, transform=val_tf)
full_test_ds  = datasets.ImageFolder(DATA_ROOT, transform=val_tf)

train_ds = Subset(full_train_ds, idx_train)
val_ds   = Subset(full_val_ds, idx_val)
test_ds  = Subset(full_test_ds, idx_test)

# ----------------------------
# Class counts (train) for sampler + optional class weights
# ----------------------------
train_targets = targets[idx_train]
class_counts = np.bincount(train_targets, minlength=num_classes)
print("\nTrain class counts:", {class_names[i]: int(class_counts[i]) for i in range(num_classes)})

# WeightedRandomSampler (balances classes per batch)
class_weights_for_sampling = 1.0 / np.clip(class_counts, 1, None)
sample_weights = class_weights_for_sampling[train_targets]
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).double(),
    num_samples=len(sample_weights),
    replacement=True
)

# Class weights for loss (optional)
class_weights_loss = (class_counts.sum() / np.clip(class_counts, 1, None))
class_weights_loss = class_weights_loss / class_weights_loss.mean()
class_weights_t = torch.tensor(class_weights_loss, dtype=torch.float32).to(device)

# ----------------------------
# DataLoaders
# ----------------------------
train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                          num_workers=cfg["num_workers"], pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"], pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"], pin_memory=True)

# ----------------------------
# Model: EfficientNet-B3 (pretrained) + new classifier head
# ----------------------------
weights = EfficientNet_B3_Weights.DEFAULT
model = efficientnet_b3(weights=weights)
in_feats = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_feats, num_classes)
model = model.to(device)

# ----------------------------
# MixUp utilities
# ----------------------------
def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------
# Loss / Optimizer / Scheduler
# ----------------------------
if cfg["loss"]["use_class_weights"]:
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg["scheduler"]["tmax"], eta_min=cfg["scheduler"]["min_lr"]
)

# ----------------------------
# Train / Eval loops
# ----------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_true = [], []
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)
    acc = correct / max(total, 1)
    avg_loss = running_loss / max(total, 1)
    return avg_loss, acc, all_true, all_preds

def train_one_epoch(epoch):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # MixUp
        use_mix = (cfg["mixup"]["alpha"] > 0) and (random.random() < cfg["mixup"]["prob"])
        optimizer.zero_grad(set_to_none=True)

        if use_mix:
            x_m, y_a, y_b, lam = mixup_data(x, y, alpha=cfg["mixup"]["alpha"])
            logits = model(x_m)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            preds = logits.argmax(dim=1)
            # "soft" accuracy approximation: count as correct if matches either label (rough)
            correct += (preds == y).sum().item()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += y.numel()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

# ----------------------------
# Plot helpers (matplotlib without styling)
# ----------------------------
def save_curve(xs, ys_dict, title, xlabel, ylabel, out_path):
    plt.figure()
    for k, v in ys_dict.items():
        plt.plot(xs, v, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_confmat(cm, labels, out_path, normalize=False):
    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # annotate
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() * 0.6 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------------------
# Training with early stopping
# ----------------------------
out_dir = Path(cfg["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "used_config.yaml", "w") as f:
    yaml.safe_dump({"cnn": cfg}, f, sort_keys=False)
    
history = {
    "epoch": [],
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
    "lr": []
}

best_val_acc = -1.0
best_epoch = -1
pat = 0

t0 = time.time()
for epoch in range(1, cfg["epochs"] + 1):
    tr_loss, tr_acc = train_one_epoch(epoch)
    va_loss, va_acc, _, _ = evaluate(val_loader)

    lr_now = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["epoch"].append(epoch)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    history["lr"].append(lr_now)

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:3d}/{cfg["epochs"]} | TrainAcc {tr_acc:.3f} ValAcc {va_acc:.3f} | LR {lr_now:.2e}")

    # Save best
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_epoch = epoch
        pat = 0
        torch.save(model.state_dict(), out_dir / "best_model.pt")
    else:
        pat += 1

    # Always save last
    torch.save(model.state_dict(), out_dir / "last_model.pt")

    # Early stop
    if pat >= cfg["early_stopping"]["patience"]:
        print(f"⏹ Early stopping at epoch {epoch}, best epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
        break

t_train = (time.time() - t0) / 60
print(f"✅ Training done in {t_train:.2f} min")
print(f"✅ Best val acc: {best_val_acc:.6f} at epoch {best_epoch}")

# Save history.csv
hist_df = pd.DataFrame(history)
hist_csv = out_dir / "history.csv"
hist_df.to_csv(hist_csv, index=False)
print("✅ Saved history:", hist_csv)

# ----------------------------
# Load best model and evaluate on TEST
# ----------------------------
model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

test_loss, test_acc, y_true, y_pred = evaluate(test_loader)

macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

print("\nTEST acc:", test_acc)
print("TEST macro P/R/F1:", macro[:3])
print("TEST weighted P/R/F1:", weighted[:3])

report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
print("\nTEST classification report:\n", report)

# Save report
(out_dir / "classification_report.txt").write_text(report)

# Per-class metrics CSV
per_p, per_r, per_f1, per_s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
per_df = pd.DataFrame({
    "class": class_names,
    "precision": per_p,
    "recall": per_r,
    "f1": per_f1,
    "support": per_s
})
per_df.to_csv(out_dir / "per_class_metrics.csv", index=False)

# Confusion matrices
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
save_confmat(cm, class_names, out_dir / "confusion_matrix.png", normalize=False)
save_confmat(cm, class_names, out_dir / "confusion_matrix_normalize.png", normalize=True)

# Curves
xs = hist_df["epoch"].values
save_curve(xs,
           {"train": hist_df["train_loss"].values, "val": hist_df["val_loss"].values},
           "Loss Curve", "Epoch", "Loss", out_dir / "loss_curve.png")

save_curve(xs,
           {"train": hist_df["train_acc"].values, "val": hist_df["val_acc"].values},
           "Accuracy Curve", "Epoch", "Accuracy", out_dir / "acc_curve.png")

# LR curve
save_curve(xs,
           {"lr": hist_df["lr"].values},
           "Learning Rate Curve", "Epoch", "LR", out_dir / "lr_curve.png")

# F1 curve (use val predictions each epoch would be expensive; approximate using test final)
# We'll compute a single F1 point for display; better is compute per epoch on val, but costly.
# Here we generate a simple bar plot for per-class F1 and a macro/weighted summary curve-like figure.

plt.figure(figsize=(10, 4))
plt.bar(per_df["class"], per_df["f1"])
plt.title("Per-class F1 (TEST)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(out_dir / "f1_curve.png", dpi=200)
plt.close()

# Macro/weighted PR "snapshot" figure
plt.figure()
plt.plot([0, 1], [macro[0], macro[0]], label=f"Macro Precision {macro[0]:.3f}")
plt.plot([0, 1], [macro[1], macro[1]], label=f"Macro Recall {macro[1]:.3f}")
plt.plot([0, 1], [macro[2], macro[2]], label=f"Macro F1 {macro[2]:.3f}")
plt.plot([0, 1], [weighted[0], weighted[0]], label=f"Weighted Precision {weighted[0]:.3f}")
plt.plot([0, 1], [weighted[1], weighted[1]], label=f"Weighted Recall {weighted[1]:.3f}")
plt.plot([0, 1], [weighted[2], weighted[2]], label=f"Weighted F1 {weighted[2]:.3f}")
plt.ylim(0, 1)
plt.title("Macro/Weighted P-R-F1 (TEST snapshot)")
plt.xlabel("dummy")
plt.ylabel("score")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "macro_weighted_pr_curve.png", dpi=200)
plt.close()

# ----------------------------
# Zip outputs for download
# ----------------------------
zip_path = Path("/kaggle/working/leaf_cnn_runs.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for fp in out_dir.rglob("*"):
        if fp.is_file():
            z.write(fp, arcname=str(fp.relative_to(out_dir.parent)))

print("\n✅ Saved files in:", out_dir)
print(sorted([p.name for p in out_dir.iterdir()]))
print("\nDataset:", cfg["dataset"]["name"])
print("Source:", cfg["dataset"]["source"])
print("Root:", DATA_ROOT)
print("✅ Zipped:", zip_path)
