# Urgent Message Detection – ML Classification Project

This project implements a complete Machine Learning pipeline to classify short text messages as **Urgent** or **Normal** using TF-IDF vectorization and classical ML models.  
Five different models are trained, compared, and the best model is automatically selected using the **F1-score for the Urgent class**.

An inference script is provided to test new messages from the command line.

---

## 1. Overview

This project includes:

- Custom labelled dataset  
- TF-IDF vectorization (unigrams + bigrams)  
- Training 5 classical ML models  
- Performance comparison using Accuracy, Precision, Recall, and F1-score  
- Best-model selection  
- Explainability tools (top keywords, confusion matrix)  
- CLI-based inference  
- Config-driven modular architecture  

All trained models and metrics are saved inside the `models/` directory.

---

## 2. Project Features

- Custom dataset of labelled messages  
- TF-IDF (1-2 grams) for feature extraction  
- Models trained:
  - Logistic Regression  
  - Multinomial Naive Bayes  
  - Linear SVM  
  - SGDClassifier  
  - Random Forest  
- Automatic model comparison  
- Selection of best model based on F1(Urgent)  
- Explainability outputs:
  - Top urgent TF-IDF features  
  - Confusion matrix  
  - Model comparison bar charts  
- CLI inference support  
- Config-based project design  

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

### Optional: Create virtual environment
```bash
python -m venv .venv
```

### Activate environment (Windows)
```bash
.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Training the Model

Run the training script:

```bash
python src/train.py
```

This will:

- Load the dataset  
- Train all 5 ML models  
- Compare model metrics  
- Select the best-performing model  
- Save the following in `models/`:
  - best_model.pkl  
  - metrics.json  
  - all_metrics.json  
  - model_comparison.csv  
  - best_model_report.txt  
  - top_features_urgent.txt  

---

## 6. Testing New Messages (Inference)

Run inference:

```bash
python src/infer.py "Server is down for all users" "Let's meet tomorrow"
```

Example output:

```
[Urgent] Server is down for all users
[Normal] Let's meet tomorrow
```

The script automatically loads the best model and TF-IDF vectorizer.

---

## 7. Visualizations (Optional)

Generate charts and confusion matrix:

```bash
python src/visualize.py
```

You will see an interactive menu:

```
1. Model comparison
2. Confusion matrix
3. Top urgent keywords
```

Charts help in presentations, evaluation, and explanation.

---

## 8. Configuration

Settings stored in `config.yaml` include:

- Path to dataset  
- Text and label column names  
- Train-test split ratio  
- Positive class  
- Paths for saving model outputs  
- List of models to train  

This design makes the project modular and easy to modify.

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
- Best model is selected purely based on F1-score for the Urgent class.  
- All evaluation results are stored in the `models/` directory.  
- The project uses a clean, professional, config-driven architecture.

---

## 11. Author

Harivardhan  
Urgent Message Detection – ML Classification Project

---
