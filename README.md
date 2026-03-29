English below

# 画像品質分類 — ロゴ汚れ検出

黒背景中央の円形ロゴに付着した汚れ・異物を検出し、製品画像を **good / bad** に自動分類するシステム。

---

## ディレクトリ構成

```
2wins_AIML_test/
├── data/
│   └── raw/
│       ├── good/        # 正常品画像
│       └── bad/         # 不良品画像
├── models/
│   ├── best_model.pth           # 分類モデルの重み
│   ├── class_names.json         # クラス名・入力解像度
│   ├── patchcore_memory_bank.npy  # PatchCore メモリバンク
│   └── patchcore_config.json    # PatchCore 閾値設定
├── outputs/
│   ├── confusion_matrix.png             # 分類モデルの混同行列（行=真ラベル、列=予測ラベル）
│   ├── anomaly_confusion_matrix.png     # アンサンブルの混同行列
│   ├── misclassified.csv        # 分類モデルの誤分類リスト
│   └── anomaly_misclassified.csv  # アンサンブルの誤分類リスト
├── src/
│   ├── dataset.py               # データ読み込み・前処理
│   ├── model.py                 # ResNet-18 モデル定義
│   ├── anomaly.py               # PatchCore 実装
│   ├── train.py                 # 分類モデル学習
│   ├── train_anomaly.py         # PatchCore 学習・アンサンブル閾値調整
│   ├── inference.py             # 分類モデル推論
│   ├── inference_anomaly.py     # アンサンブル推論
│   └── utils.py                 # 混同行列・レポート出力
├── report.md                    # 手法・結果レポート（日本語）
└── README.md
```

---

## セットアップ

```bash
pip install torch torchvision scikit-learn tqdm matplotlib pillow
```

GPU不要。CPU環境で動作します。

---

## 使い方

### 方式1：分類モデルのみ（推奨・高速）

**学習：**
```bash
cd 2wins_AIML_test
python src/train.py
```
学習済みモデルは `models/best_model.pth` に保存されます。

**推論：**
```bash
# フォルダ全体
python src/inference.py --input data/raw

# 結果をCSVに保存
python src/inference.py --input data/raw --output results.csv

# 閾値を変更（デフォルト 0.3）
python src/inference.py --input data/raw --bad-threshold 0.4
```

---

### 方式2：アンサンブル（PatchCore + 分類モデル）

> **前提：** 方式1の学習（`train.py`）を先に完了させること。

**学習・閾値調整：**
```bash
python src/train_anomaly.py
```

閾値を手動指定する場合：
```bash
python src/train_anomaly.py --clf-threshold 0.5 --veto-threshold 0.05
```

**推論：**
```bash
python src/inference_anomaly.py --input data/raw

# 結果をCSVに保存
python src/inference_anomaly.py --input data/raw --output results.csv
```

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `outputs/misclassified.csv` | 分類モデルの誤分類（filepath, true_label, predicted_label, p_bad） |
| `outputs/anomaly_misclassified.csv` | アンサンブルの誤分類（+ anomaly_score, p_bad） |
| `outputs/confusion_matrix.png` | 分類モデルの混同行列。行が真のラベル（good/bad）、列が予測ラベルを表す。対角成分が正解数、非対角成分が誤分類数。 |
| `outputs/anomaly_confusion_matrix.png` | アンサンブル（PatchCore＋分類モデル）の混同行列。見方は同上。 |

---

## 主な結果

| 手法 | bad→good（見逃し） | good→bad（誤検出） | 合計（1350枚） |
|---|---|---|---|
| 分類モデル（threshold=0.3） | 3* | 11 | 14 |
| アンサンブル（veto付き） | 1* | 19 | 20 |

*1件はラベル誤りの疑いあり

**運用上の推奨：** 推論速度の制約から `inference.py`（分類モデル）を採用。

---

## 閾値チューニング

`inference.py` の `--bad-threshold`（デフォルト 0.3）を変更することで、
bad recall と false positive のトレードオフを調整できます。

| 閾値 | 傾向 |
|---|---|
| 低い（例: 0.2） | bad見逃しが減る、誤検出が増える |
| 高い（例: 0.5） | 誤検出が減る、bad見逃しが増える |

アンサンブルの場合は `models/patchcore_config.json` の
`veto_threshold` を直接編集して `inference_anomaly.py` で即時反映できます。

---

# Image Quality Classification — Logo Contamination Detection

A system that detects dirt and foreign matter on circular logos centred on a black background, automatically classifying product images as **good / bad**.

---

## Directory Structure

```
2wins_AIML_test/
├── data/
│   └── raw/
│       ├── good/        # Normal product images
│       └── bad/         # Defective product images
├── models/
│   ├── best_model.pth           # Classifier model weights
│   ├── class_names.json         # Class names and input resolution
│   ├── patchcore_memory_bank.npy  # PatchCore memory bank
│   └── patchcore_config.json    # PatchCore threshold config
├── outputs/
│   ├── confusion_matrix.png             # Classifier confusion matrix (rows=true, cols=predicted)
│   ├── anomaly_confusion_matrix.png     # Ensemble confusion matrix
│   ├── misclassified.csv        # Classifier misclassification list
│   └── anomaly_misclassified.csv  # Ensemble misclassification list
├── src/
│   ├── dataset.py               # Data loading and preprocessing
│   ├── model.py                 # ResNet-18 model definition
│   ├── anomaly.py               # PatchCore implementation
│   ├── train.py                 # Classifier training
│   ├── train_anomaly.py         # PatchCore training and ensemble threshold tuning
│   ├── inference.py             # Classifier inference
│   ├── inference_anomaly.py     # Ensemble inference
│   └── utils.py                 # Confusion matrix and report output
├── report.md                    # Method and results report (Japanese)
└── README.md
```

---

## Setup

```bash
pip install torch torchvision scikit-learn tqdm matplotlib pillow
```

No GPU required. Runs on CPU.

---

## Usage

### Method 1: Classifier only (recommended — fast)

**Training:**
```bash
cd 2wins_AIML_test
python src/train.py
```
The trained model is saved to `models/best_model.pth`.

**Inference:**
```bash
# Entire folder
python src/inference.py --input data/raw

# Save results to CSV
python src/inference.py --input data/raw --output results.csv

# Change threshold (default 0.3)
python src/inference.py --input data/raw --bad-threshold 0.4
```

---

### Method 2: Ensemble (PatchCore + Classifier)

> **Prerequisite:** Complete Method 1 training (`train.py`) first.

**Training and threshold tuning:**
```bash
python src/train_anomaly.py
```

To specify thresholds manually:
```bash
python src/train_anomaly.py --clf-threshold 0.5 --veto-threshold 0.05
```

**Inference:**
```bash
python src/inference_anomaly.py --input data/raw

# Save results to CSV
python src/inference_anomaly.py --input data/raw --output results.csv
```

---

## Output Files

| File | Contents |
|---|---|
| `outputs/misclassified.csv` | Classifier misclassifications (filepath, true_label, predicted_label, p_bad) |
| `outputs/anomaly_misclassified.csv` | Ensemble misclassifications (+ anomaly_score, p_bad) |
| `outputs/confusion_matrix.png` | Classifier confusion matrix. Rows = true labels (good/bad), columns = predicted labels. Diagonal = correct predictions, off-diagonal = misclassifications. |
| `outputs/anomaly_confusion_matrix.png` | Ensemble (PatchCore + classifier) confusion matrix. Same layout as above. |

---

## Key Results

| Method | bad→good (missed) | good→bad (false positive) | Total (1350 images) |
|---|---|---|---|
| Classifier (threshold=0.3) | 3* | 11 | 14 |
| Ensemble (with veto) | 1* | 19 | 20 |

*1 case is suspected mislabelled

**Operational recommendation:** Due to inference speed constraints, `inference.py` (classifier only) is adopted.

---

## Threshold Tuning

Adjust the `--bad-threshold` in `inference.py` (default 0.3) to control the trade-off between bad recall and false positives.

| Threshold | Effect |
|---|---|
| Lower (e.g. 0.2) | Fewer missed bad samples, more false positives |
| Higher (e.g. 0.5) | Fewer false positives, more missed bad samples |

For the ensemble, edit `veto_threshold` directly in `models/patchcore_config.json` and re-run `inference_anomaly.py` to apply immediately.
