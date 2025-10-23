# Rock–Paper–Scissors CNN

This project trains Convolutional Neural Networks (CNNs) to classify hand gestures for **rock**, **paper**, and **scissors**.

We focus on:
- Using proper splits (train / validation / test) with **no test leakage**
- Preprocessing: **resize + normalization**, and **train-only** augmentation
- Designing **three** CNN architectures (Tiny / Base / Deep) and comparing them
- **Automatic** hyperparameter tuning with cross-validation (macro-F1)
- Final **one-shot** test evaluation with full metrics

---

## Open Notebook

- **Main notebook:** `main.ipynb`  
- *(Optional Colab link – replace USERNAME/REPO if you want this button)*
  - [Open in Colab](https://colab.research.google.com/github/USERNAME/REPO/blob/main/main.ipynb)

---

## Project Structure

rps-cnn/
├─ main.ipynb # run top→bottom (has all steps)
├─ requirements.txt # Python dependencies
├─ .gitignore # excludes data/.venv/checkpoints/logs
├─ splits/ # fixed file lists (created by the notebook)
│ ├─ train.txt
│ ├─ val.txt
│ └─ test.txt
├─ reports/ # figures & (optionally) your PDF
│ ├─ TinyCNN_acc.png
│ ├─ TinyCNN_loss.png
│ ├─ BaseCNN_acc.png
│ ├─ BaseCNN_loss.png
│ ├─ DeepCNN_acc.png
│ ├─ DeepCNN_loss.png
│ └─ test_confusion_matrix.png
└─ data/ # local only (ignored by Git)
└─ raw/
└─ paper | rock | scissors

---

## Dataset

- Kaggle: **Rock–Paper–Scissors**  
  https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data

**Place locally (not in the repo):**
