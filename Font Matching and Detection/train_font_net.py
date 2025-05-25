import json
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


class FontCropDataset(Dataset):
    def __init__(self, root: Path, img_size: int = 96) -> None:
        super().__init__()
        self.img_dir = root / "images"
        self.lbl_dir = root / "labels"
        self.paths = list(self.img_dir.glob("*.jpg"))
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.id_to_font = json.loads((root / "font_label_map.json").read_text())

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape

        class_id, cx, cy, bw, bh = map(float, lbl_path.read_text().splitlines()[0].split())
        cx, cy, bw, bh = cx * W, cy * H, bw * W, bh * H
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        crop = img[y1:y2, x1:x2]
        crop = self.transform(crop)
        return crop, int(class_id)


class FontNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(data_root: Path, epochs: int, batch: int, img_size: int, weights: Path) -> None:
    full_ds = FontCropDataset(data_root, img_size)
    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    device = get_device()
    print(f"Using device: {device}")

    model = FontNet(n_classes=len(full_ds.id_to_font)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"validation accuracy: {acc:.3f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), weights)

    print(f"best accuracy: {best:.3f}  |  model saved to {weights}")


if __name__ == "__main__":
    data_root = Path("dataset")
    epochs    = 25
    batch     = 64
    img_size  = 96
    weights   = Path("font_net.pth")

    train(data_root, epochs, batch, img_size, weights)
