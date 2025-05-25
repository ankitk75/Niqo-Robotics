import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms


class FontNet(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 12 * 12, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_hello_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    dil = cv2.dilate(th, kernel, iterations=2)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w / h > 2 and w > 50 and h > 15:
            boxes.append((x, y, w, h))
    return boxes


def classify_crop(model, crop_gray, tfm, device):
    with torch.no_grad():
        t = tfm(crop_gray).unsqueeze(0).to(device)
        logits = model(t)
        prob = torch.softmax(logits, 1)[0]
        cls = prob.argmax().item()
        conf = prob.max().item()
    return cls, conf


def run_detection(image_path: Path,
                  weights_path: Path,
                  label_map_path: Path,
                  out_path: Path) -> None:
    device = get_device()
    print(f"Using device: {device}")
    id_to_font = json.loads(label_map_path.read_text())
    model = FontNet(n_classes=len(id_to_font))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    results = []
    for (x, y, w, h) in find_hello_regions(img):
        crop = img[y:y+h, x:x+w]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cls, conf = classify_crop(model, crop_gray, tfm, device)
        font_name = id_to_font[str(cls)]
        results.append({"bbox": [int(x), int(y), int(w), int(h)],
                        "font": font_name,
                        "confidence": round(conf, 3)})
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{font_name} {conf:.2f}",
                    (x, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)
    cv2.imwrite(str(out_path), img)
    print(json.dumps({"detectedFonts": results}, indent=2))


if __name__ == "__main__":
    IMAGE_PATH   = Path("test_page.png")
    WEIGHTS_PATH = Path("font_net.pth")
    LABEL_MAP    = Path("dataset/font_label_map.json")
    OUT_PATH     = Path("annotated.jpg")
    run_detection(IMAGE_PATH, WEIGHTS_PATH, LABEL_MAP, OUT_PATH)
