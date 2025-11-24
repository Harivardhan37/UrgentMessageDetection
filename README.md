# Urgent Message Detection – ML Classification Project

## 1. Overview
This project builds a Machine Learning pipeline to classify short text messages as **Urgent** or **Normal**.  
It uses TF-IDF features and evaluates **multiple classical ML classifiers**, selecting the best based on F1-score for the urgent class.

The final model is saved as `best_model.pkl`, and an inference script is provided to test new messages.

---

## 2. Project Features
- Custom dataset of labelled messages  
- TF-IDF text vectorization (uni + bi-grams)  
- Training of **5 different classifiers**:
  - Logistic Regression  
  - Multinomial Naive Bayes  
  - Linear SVM  
  - SGDClassifier  
  - Random Forest  
- Automatic model comparison using Accuracy, Precision, Recall, and F1-score  
- Best model selection  
- Explainability:
  - Top TF-IDF features for “Urgent”  
  - Confusion matrix  
  - Model comparison bar charts  
- Clean config-based project structure  
- CLI inference support  

---

## 3. Project Structure
```
urgent-message-detection/
│
├── data/
│   └── messages.csv
│
├── models/
│   ├── best_model.pkl
│   ├── metrics.json
│   ├── all_metrics.json
│   ├── model_comparison.csv
│   ├── best_model_report.txt
│   └── top_features_urgent.txt
│
├── src/
│   ├── train.py
│   ├── infer.py
│   └── visualize.py
│
├── config.yaml
├── requirements.txt
├── README.md
└── REPORT.md
```

---

## 4. Installation

### 4.1 Create virtual environment (optional)
```bash
python -m venv .venv
```

### 4.2 Activate it
**Windows:**
```bash
.venv\Scripts\activate
```

### 4.3 Install dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Training the Model
Run:

```bash
python src/train.py
```

This will:

- Load dataset  
- Train all 5 ML models  
- Compare metrics  
- Select best model  
- Save:
  - `models/best_model.pkl`
  - `models/metrics.json`
  - `models/all_metrics.json`
  - `models/model_comparison.csv`
  - `models/best_model_report.txt`
  - `models/top_features_urgent.txt`

---

## 6. Testing New Messages (Inference)
To classify new text messages:

```bash
python src/infer.py "Server is down for all users" "Let's meet tomorrow"
```

Example output:

```
[Urgent] Server is down for all users
[Normal] Let's meet tomorrow
```

The inference script automatically loads the best saved model.

---

## 7. Visualizations (Optional)
To view comparison charts, confusion matrix, or top urgent keywords:

```bash
python src/visualize.py
```

You will get an interactive menu:

```
1. Model comparison
2. Confusion matrix
3. Top urgent keywords
```

These visuals are useful for analysis or presentation in the report.

---

## 8. Configuration
Settings are stored in `config.yaml`:

- dataset path  
- text/label column names  
- test split  
- positive label  
- model save paths  
- list of models used  

This makes the project clean and configurable.

---

## 9. Requirements
```
pandas
scikit-learn
pyyaml
joblib
matplotlib
```

---

## 10. Notes
- The dataset (`messages.csv`) is manually created and balanced.  
- The best model is selected purely based on F1-score for the **Urgent** class.  
- All evaluation results are automatically exported under the `models/` directory.

---

## 11. Author
Harivardhan  
Urgent Message Detection ML Assignment