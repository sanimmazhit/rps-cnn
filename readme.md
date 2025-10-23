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


---

## Project Structure

rps-cnn/
- `main.ipynb` — main code
- `requirements.txt` — dependencies
- `.gitignore` — ignores data/.venv/checkpoints/logs
- `splits/` — `train.txt`, `val.txt`, `test.txt`
- `reports/` — training curves + `test_confusion_matrix.png`
- `data/raw/` — local dataset 


---

## Dataset

- Kaggle: **Rock–Paper–Scissors**  
  https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data

**Place locally (not in the repo):**
