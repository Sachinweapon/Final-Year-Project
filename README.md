# UPI Fraud Detection Using Machine Learning

A machine learning system to detect fraudulent UPI (Unified Payments Interface) transactions in real-time, built as a Final Year Project.

---

## 📌 Problem Statement

With the rapid adoption of UPI as a payment platform in India, fraudulent transactions have become a growing concern. Traditional rule-based systems fail to adapt to evolving fraud patterns. This project applies machine learning to classify transactions as genuine or fraudulent based on historical data.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** XGBoost, Scikit-learn, Pandas, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook (Anaconda)

---

## 📂 Project Structure

```
Final-Year-Project/
├── UPI_Fraud_Project/
│   ├── data_processing.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
│   └── data/
│       ├── transactions_train.csv
│       └── transactions_test.csv
└── Research Paper/             # Reference research paper
```

---

## 📊 Dataset

- **Source:** [Kaggle](https://www.kaggle.com/)
- **Size:** ~100,000 transactions (train + test split)
- **Class Distribution:** ~98% genuine, ~2% fraudulent (~60:1 imbalance)
- **Key Features:** `transaction_amount`, `payment_channel`, `device_type`, `hour`, `day_of_week`

---

## ⚙️ Methodology

1. **Preprocessing** — Drop non-predictive ID columns, extract time features (hour, day of week), label encode categorical columns
2. **Imbalance Handling** — Computed `scale_pos_weight = 60.6` to penalise missed fraud detections during training
3. **Model Training** — XGBoost Classifier with 500 estimators, max depth 6, learning rate 0.05
4. **Threshold Tuning** — Decision threshold set to `0.3` (instead of default 0.5) to prioritise recall on the fraud class
5. **Evaluation** — Classification report, confusion matrix, ROC-AUC, Precision-Recall curve

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | ~100% |
| Fraud Precision | 0.98 |
| Fraud Recall | 0.98 |
| Fraud F1-Score | 0.98 |
| ROC-AUC | 0.9999 |

**Confusion Matrix (Test Set):**
|  | Predicted: Not Fraud | Predicted: Fraud |
|---|---|---|
| **Actual: Not Fraud** | 97,862 | 42 |
| **Actual: Fraud** | 30 | 1,953 |

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/Sachinweapon/Final-Year-Project.git
   ```
2. Install dependencies
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn jupyter
   ```
3. Place the dataset CSVs inside `UPI_Fraud_Project/data/`
4. Open and run the notebook
   ```bash
   jupyter notebook UPI_Fraud_Project/data_processing.ipynb
   ```

---

## 🔍 Model Comparison

Three models were evaluated on fraud recall:

| Model | Recall (Fraud) |
|---|---|
| Logistic Regression | Lower |
| Random Forest | Moderate |
| **XGBoost** | **Highest** |

XGBoost was selected as the final model due to its superior recall and ability to handle class imbalance natively.
