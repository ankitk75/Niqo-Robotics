# 1. Font-Matching & Detection  


Explanation of how **exactly how to execute each stage** of the
pipeline **and** documents **why every neural-network layer is present**.

---
## Workflow

| Step | Command | Output |
|------|---------|--------|
| 1. Generate synthetic pages | `python generate_dataset.py` | `dataset/` ( 10,000 pages + labels) |
| 2. Train classifier | `python train_font_net.py` | `font_net.pth` |
| 4. Detect & classify | *(edit paths at bottom of `detect_and_classify.py` then run)*<br>`python detect_and_classify.py` | `annotated.jpg` + JSON |

---

## CNN – Layer Purpose (brief)

| Layer / Block | Reason |
|---------------|--------|
| **Conv 3 × 3 / 32 + BatchNorm + ReLU** | Captures strokes & edges; BN speeds convergence |
| **MaxPool 2 × 2** | Halves resolution; text images are simple |
| **Conv 3 × 3 / 64 + BN + ReLU** | Learns character-level shapes |
| **MaxPool 2 × 2** | Further down-sampling |
| **Conv 3 × 3 / 128 + BN + ReLU** | Extracts glyph-level patterns |
| **MaxPool 2 × 2** | Final spatial squeeze |
| **Flatten → FC (18 432 → 256) + ReLU** | Dense feature mixing |
| **Dropout 0.3** | Mitigates over-fitting on small data |
| **FC 256 → 10 (logits)** | 10-font classification |
| **Label-smoothing 0.1** | Reduces over-confident soft-max outputs |

### Output Image
<img src="annotated.jpg" alt="output"/>
