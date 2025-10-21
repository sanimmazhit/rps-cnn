# RPS CNN — Rock–Paper–Scissors Image Classification

A clean, reproducible ML project that trains **three CNNs** (increasing complexity) to classify hand gestures for **rock**, **paper**, and **scissors**. It follows sound methodology: fixed train/val/test splits, **no test leakage**, automatic hyperparameter tuning with **cross-validation**, full metrics (accuracy, precision, recall, **F1**), saved training curves, and a final one-shot test evaluation.

---

## 1) Objectives (what I implement)

- Build a CNN to classify **rock / paper / scissors** images.
- Use good ML practice:
  - **Exploration** of the dataset.
  - **Preprocessing**: resize + normalization; train-only augmentation.
  - **Correct split** into train/validation/test with no leakage.
- Design **3 architectures** (Tiny / Base / Deep) and compare them.
- **Hyperparameter tuning** (learning rate, dropout, batch size) via **grid + K-fold CV** on **training** only.
- Evaluate with **accuracy, precision, recall, F1** and a **confusion matrix**.
- Save training **curves** and discuss **over/underfitting**.
- *(Optional)* test generalization on my own photos.

---

## 2) Dataset

- Kaggle: Rock-Paper-Scissors  
  https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data

**Place the data locally (not in the repo):**
data/raw/
paper/
rock/
scissors/

*(If your download extracts as `rockpaperscissors/paper|rock|scissors`, put that folder under `data/raw/rockpaperscissors/`. The notebook auto-detects either layout.)*

---

## 3) Project structure

├── main.ipynb # single, commented notebook (run top→bottom)
├── requirements.txt # Python deps (Apple Silicon or Intel block)
├── .gitignore # keeps data/.venv/checkpoints out of the repo
├── splits/ # saved lists (created by the notebook)
│ ├── train.txt
│ ├── val.txt
│ └── test.txt
├── reports/
│ ├── rps_report.pdf # my 10–15 page report
│ ├── TinyCNN_acc.png # training curves (auto-saved)
│ ├── TinyCNN_loss.png
│ ├── BaseCNN_acc.png
│ ├── BaseCNN_loss.png
│ ├── DeepCNN_acc.png
│ ├── DeepCNN_loss.png
│ └── test_confusion_matrix.png
└── data/
└── raw/ # local dataset only (ignored by Git)
└── paper|rock|scissors


---

## 4) Setup

### Python (VS Code easiest)
1. Open the folder in **VS Code**.
2. Press **⌘⇧P** → **Python: Create Environment** → **Venv** → select `requirements.txt`.
3. Open `main.ipynb` → top-right kernel picker → select the **.venv** interpreter from this project.

### `requirements.txt`
Choose **one** block (don’t mix them):

**Apple Silicon (M1/M2/M3)**
numpy
pandas
matplotlib
scikit-learn
tensorflow-macos
tensorflow-metal
ipykernel

**Intel**
numpy
pandas
matplotlib
scikit-learn
tensorflow==2.16.*
ipykernel

---

## 5) How to run

1. Put the dataset under `data/raw/…` (see layout above).
2. Open `main.ipynb`. At the top:
   - `QUICK_RUN = True` (fast debug), `RUN_FINAL_TEST = False`.
3. Run the notebook **top → bottom** through the **tuning cell**:
   - Creates **fixed splits** (`splits/*.txt`)
   - Trains **Tiny / Base / Deep** and saves **curves**
   - Does **grid + K-fold CV** (training only) and prints **top configs**
4. For the **proper run**:
   - Set `QUICK_RUN = False`, re-run through tuning.
5. Final one-shot test:
   - Set `RUN_FINAL_TEST = True`, run the **final evaluation** cell to print metrics and save the **confusion matrix** PNG.

**Speed knobs:**  
- `QUICK_RUN=True` → smaller image size, fewer epochs/folds/grid points.  
- `QUICK_RUN=False` → full training/tuning for the report.

---

## 6) What the notebook does (by section)

- **EDA:** index all images, class counts, show a few samples.
- **Preprocessing:** resize to 128×128 (or 150×150 in full run), normalize to [0,1], train-only augmentation (flip/rotation/zoom).
- **Splits:** stratified **train/val/test**; splits are **saved** to disk and reused.
- **Models:** three CNNs with increasing depth; BatchNorm + Dropout; softmax output.
- **Training:** Adam, cross-entropy, **EarlyStopping** + **ReduceLROnPlateau**; curves saved to `reports/`.
- **Tuning:** Base model hyperparameters with **grid + Stratified K-fold**; select by **macro-F1** (not just accuracy).
- **Final eval:** retrain best on **train+val**, evaluate **once** on **test**; print **classification report** and **confusion matrix** (saved as PNG).
- **Optional:** generalization on my own photos in `data/my_photos/...`.

---

## 7) Results (example placeholders)

*(I replace these with my actual numbers when I run the notebook.)*

**Validation (best of Tiny/Base/Deep):**
| Model    | Val Acc | Notes |
|----------|---------|-------|
| TinyCNN  | 0.88    | stable, lower capacity |
| BaseCNN  | 0.93    | best speed/accuracy trade-off |
| DeepCNN  | 0.94    | slightly higher, watch overfitting |

**Tuning (top configs by mean macro-F1):**
[
{"lr": 0.001, "dr1": 0.4, "batch": 32, "mean_macro_f1": 0.95},
...
]

**Final test (one-shot):**
- Accuracy: …
- Macro-F1: …
- Per-class precision/recall/F1: see the printed classification report.
- Confusion matrix: `reports/test_confusion_matrix.png`

**Over/underfitting notes:**  
- I compare training vs validation curves (acc/loss) and describe where the gap grows/shrinks, and how dropout/augmentation affected it.

---

## 8) Reproducibility

- Fixed **seeds** (NumPy/TensorFlow; deterministic ops when available).
- **Saved splits** (`splits/*.txt`) used across runs (prevents test leakage).
- Exact environment in `requirements.txt`.
- Clear run instructions (quick vs full).
- Code and report in a **public GitHub** repository (dataset excluded).

---

## 9) Report (PDF)

I submit **`reports/rps_report.pdf`** (10–15 pages) including:
- **Declaration** (required text from the assignment).
- Dataset & EDA; preprocessing; split policy.
- Model architectures with justification.
- Training curves (with commentary on over/underfitting).
- Tuning method & grid; CV results; chosen hyperparams.
- Final test metrics (accuracy, precision, recall, F1); confusion matrix; misclassification analysis.
- Optional generalization results.
- Reproducibility and repo link.

---

## 10) Don’t commit large/local stuff

This repo ignores:
data/raw/
data/processed/
.venv/
.ipynb_checkpoints/
checkpoints/
logs/
pycache/
*.pyc
.DS_Store

---

## 11) Acknowledgments

- Dataset by **Laurence Moroney** on Kaggle (Rock–Paper–Scissors).

