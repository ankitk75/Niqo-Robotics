import argparse
import json
import random
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder

def rectangles_intersect(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def load_seed_images(src_dir: Path):
    dataset = ImageFolder(root=str(src_dir))
    samples = []
    for path, cls in dataset.imgs:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            samples.append((img, cls))
    return samples, dataset.classes

def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    scale = rng.uniform(0.8, 1.2)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    angle = rng.uniform(-5, 5)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        noise = rng.normalvariate(0, 10)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def paste_crop(canvas, crop, x, y):
    h, w = crop.shape
    canvas[y : y + h, x : x + w] = crop
    H, W = canvas.shape
    cx, cy, bw, bh = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
    rect = (x, y, x + w, y + h)
    return (cx, cy, bw, bh), rect

def generate_dataset(src_dir: Path,
                     out_dir: Path,
                     n_pages: int,
                     max_per_page: int,
                     seed: int = 42):
    rng = random.Random(seed)

    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    for d in (img_out, lbl_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    seeds, class_names = load_seed_images(src_dir)
    if not seeds:
        print("No seed images found.")
        return

    CANVAS_W, CANVAS_H = 800, 600
    MAX_TRIES = 30

    for _ in tqdm(range(n_pages), desc="Generating"):
        canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        labels, placed_rects = [], []

        n_phrases = rng.randint(1, max_per_page)
        for _ in range(n_phrases):
            crop, cls = rng.choice(seeds)
            crop = augment(crop.copy(), rng)
            scale = min((CANVAS_W * 0.8) / crop.shape[1],
                        (CANVAS_H * 0.8) / crop.shape[0], 1.0)
            if scale < 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
            h, w = crop.shape

            valid_pos = None
            for _ in range(MAX_TRIES):
                x = rng.randint(10, CANVAS_W - w - 10)
                y = rng.randint(10, CANVAS_H - h - 10)
                rect = (x, y, x + w, y + h)
                if all(not rectangles_intersect(rect, r) for r in placed_rects):
                    valid_pos = (x, y, rect)
                    break

            if valid_pos is None:
                continue

            x, y, rect = valid_pos
            yolo_box, _ = paste_crop(canvas, crop, x, y)
            labels.append(f"{cls} {' '.join(f'{v:.6f}' for v in yolo_box)}")
            placed_rects.append(rect)

        uid = uuid.uuid4().hex
        cv2.imwrite(str(img_out / f"{uid}.jpg"), canvas)
        (lbl_out / f"{uid}.txt").write_text("\n".join(labels))

    mapping = {i: name for i, name in enumerate(class_names)}
    (out_dir / "font_label_map.json").write_text(json.dumps(mapping, indent=2))
    print(f"Dataset written to {out_dir}")

if __name__ == "__main__":
    SRC_DIR       = Path("'/Users/ankitkumar/Desktop/Niqo/Font Matching and Detection/images'")
    OUT_DIR       = Path("dataset")
    N_PAGES       = 10000
    MAX_PER_PAGE  = 4
    SEED          = 42

    generate_dataset(
        src_dir=SRC_DIR,
        out_dir=OUT_DIR,
        n_pages=N_PAGES,
        max_per_page=MAX_PER_PAGE,
        seed=SEED,
    )
