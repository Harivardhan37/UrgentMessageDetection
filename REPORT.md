# Urgent Message Detection – ML Project Report

## 1. Introduction
The goal of this project is to build a Machine Learning model that can automatically classify a given text message as **Urgent** or **Normal**.  
This kind of system is useful in customer support, IT operations, alerting pipelines, and messaging platforms where urgent communication needs to be prioritized instantly.

This report explains the dataset creation, preprocessing pipeline, the different ML models trained, the evaluation metrics, comparisons, and the final model selection.

---

## 2. Dataset Description
A custom labelled dataset (`messages.csv`) was created for this assignment. It includes a balanced set of **Urgent** and **Normal** messages.

### Types of messages included:
- IT/system outage alerts  
- Payment failures  
- Server-side issues  
- Health/emergency messages  
- Scheduling, personal, and day-to-day normal messages  
- Mild Telugu-flavoured English messages to increase variety  
- Campus-related examples  

This ensures diversity and reduces bias.

### Columns:
- **message** → the text  
- **label** → "Urgent" or "Normal"  

A stratified split (80% train, 20% test) was used.

---

## 3. Preprocessing
All preprocessing is handled by scikit-learn’s `TfidfVectorizer` inside a pipeline.

### Steps performed:
- Convert text to lowercase  
- Remove English stopwords  
- Extract **unigrams and bigrams**  
- Convert text into TF-IDF vectors  
- Produce a sparse, high-dimensional feature matrix  

TF-IDF is ideal for short text classification and works well with linear models.

---

## 4. Models Trained
Five classical ML models were trained and compared using identical TF-IDF features.

### 1. Logistic Regression
- Linear model  
- Strong baseline for text  
- Good precision & interpretability  

### 2. Multinomial Naive Bayes
- Probabilistic model  
- Very fast  
- Works well for word count–based classification  

### 3. Linear SVM (LinearSVC)
- Margin-based classifier  
- Often best for high-dimensional sparse text  
- Very strong generalization  

### 4. SGDClassifier (Logistic Loss)
- Linear model trained with stochastic gradient descent  
- Fast, scalable  
- Performs similarly to LR/SVM  

### 5. Random Forest
- Ensemble of decision trees  
- Included mainly for comparison  
- Generally weaker for sparse TF-IDF features  

Each model was trained using the same training split and evaluated using the same test split.

---

## 5. Evaluation Metrics
The following metrics were computed for each model:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score** (with “Urgent” as the positive class)  

The F1-score for **Urgent** is the most important metric because false negatives  
(urgent messages predicted as normal) are costly.

---

## 6. Model Comparison

The comparison table was automatically exported as `models/model_comparison.csv`.

*(Example layout shown below — actual numbers depend on runtime results)*

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | ... | ... | ... | ... |
| Naive Bayes | ... | ... | ... | ... |
| Linear SVM | ... | ... | ... | ... |
| SGDClassifier | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |

---

## 7. Best Model Selection
The **best model** was selected based on the **highest F1-score for the “Urgent” class**.

This ensures:
- Urgent messages are detected reliably  
- Fewer false negatives  
- Higher real-world safety in alerting systems  

The selected model is saved as:

```
models/best_model.pkl
```

The detailed classification report of the best model is saved as:

```
models/best_model_report.txt
```

---

## 8. Explainability – Top Urgent Keywords
For linear models (LR, Linear SVM, SGD), we extracted the **top TF-IDF features** that strongly contribute to predicting "Urgent".

These are saved in:

```
models/top_features_urgent.txt
```

These keywords typically include:
- “down”
- “failure”
- “immediately”
- “asap”
- “server”
- “error”
- “urgent”

This helps validate that the model is learning meaningful patterns.

---

## 9. Visualizations
To support analysis and presentation, `visualize.py` generates:

### ✔ Model Comparison Charts
Accuracy and F1-score bar charts for all 5 models.

### ✔ Confusion Matrix (Best Model)
Shows:
- True Urgent detected  
- False Urgent  
- False Normal  
- True Normal  

Useful for understanding mistakes.

### ✔ Top TF-IDF Features Plot
Displays the most influential urgent-related keywords.

These are optional but helpful for evaluation and interview discussion.

---

## 10. Final Results (Summary)
- A complete ML pipeline was built using TF-IDF + classical classifiers.  
- 5 models were trained and compared using standardized metrics.  
- The best model was automatically selected based on **F1(Urgent)**.  
- Explainability and visual exploration were included.  
- The project is modular, config-driven, and easy to run.  

The system predicts unseen messages using:

```
python src/infer.py "your message here"
```

---

## 11. Limitations
- Dataset size is small (hand-created)  
- TF-IDF doesn't capture deep semantic meaning  
- No hyperparameter tuning performed  
- No handling of extremely long messages  

---

## 12. Future Work
- Use larger real-world datasets  
- Implement hyperparameter optimization  
- Try deep learning models (BERT, DistilBERT)  
- Deploy the model as a REST API (FastAPI)  
- Add model monitoring and drift detection  

---

## 13. Conclusion
This project successfully demonstrates the complete workflow for urgent message detection using classical Machine Learning methods.  
The model comparison approach provides a clear understanding of which algorithm performs best for this task, and the explainability tools confirm that the model learns intuitive patterns.

This report, combined with the codebase and documentation, forms a complete, end-to-end ML assignment submission.

