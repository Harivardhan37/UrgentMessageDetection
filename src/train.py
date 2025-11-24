import os
import json
import argparse

import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_models(max_iter: int = 500):
    """
    Build multiple TF-IDF based models:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Linear SVM
    - SGDClassifier (linear model, good for text)
    - Random Forest (for contrast with tree-based model)
    """
    # Common vectorizer config for all models
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        stop_words="english",
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", LogisticRegression(max_iter=max_iter)),
            ]
        ),
        "naive_bayes": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", MultinomialNB()),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", LinearSVC()),
            ]
        ),
        "sgd_classifier": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",  # logistic regression style
                        max_iter=max_iter,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(model, X_test, y_test, pos_label: str):
    """
    Evaluate a model and return metrics + full classification report.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=pos_label
    )
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
    }


def save_top_features_urgent(model, top_k: int, output_path: str):
    """
    Save top-k features that are most indicative of the 'Urgent' class
    for linear models that expose coef_.
    """
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]

        # Only works for linear models with coef_ attribute
        if not hasattr(clf, "coef_"):
            print("[WARN] Selected best model does not expose 'coef_' for features.")
            return

        feature_names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]  # binary classification case

        top_indices = coefs.argsort()[::-1][:top_k]
        top_features = [(feature_names[i], float(coefs[i])) for i in top_indices]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Top features for 'Urgent' class:\n\n")
            for word, weight in top_features:
                f.write(f"{word}: {weight:.4f}\n")

    except Exception as e:
        print(f"[WARN] Could not save top features: {e}")


def main(config_path: str):
    # 1. Load config
    cfg = load_config(config_path)

    data_path = cfg["data"]["path"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    test_size = cfg["train"]["test_size"]
    random_state = cfg["train"]["random_state"]

    max_iter = cfg["models"]["max_iter"]
    pos_label = cfg["models"]["pos_label"]

    model_dir = cfg["paths"]["model_dir"]
    model_path = cfg["paths"]["model_path"]
    metrics_path = cfg["paths"]["metrics_path"]

    os.makedirs(model_dir, exist_ok=True)

    # 2. Load data
    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' in CSV. "
            f"Found columns: {df.columns.tolist()}"
        )

    X = df[text_col]
    y = df[label_col]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Build candidate models
    models = build_models(max_iter=max_iter)

    all_metrics = {}
    best_model_name = None
    best_model = None
    best_f1 = -1.0
    best_report = ""

    # 5. Train & evaluate each model
    print("[INFO] Training and evaluating models...")
    for name, model in models.items():
        print(f"\n[INFO] Training model: {name}")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, pos_label=pos_label)

        all_metrics[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

        print(f"--- {name} ---")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1']:.4f}")

        # Track best model by F1 for the positive (Urgent) class
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model
            best_report = metrics["report"]

    print(f"\n[INFO] Best model based on F1({pos_label}): {best_model_name} (F1 = {best_f1:.4f})")

    # 6. Save per-model metrics (JSON + CSV)
    all_metrics_path = os.path.join(model_dir, "all_metrics.json")
    with open(all_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[INFO] All model metrics saved to {all_metrics_path}")

    # Also save as CSV for easy viewing / report table
    df_metrics = pd.DataFrame.from_dict(all_metrics, orient="index")
    df_metrics.index.name = "model"
    comparison_csv_path = os.path.join(model_dir, "model_comparison.csv")
    df_metrics.to_csv(comparison_csv_path)
    print(f"[INFO] Model comparison table saved to {comparison_csv_path}")

    # Save detailed report for best model
    report_path = os.path.join(model_dir, "best_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(best_report)
    print(f"[INFO] Best model classification report saved to {report_path}")

    # 7. Save feature importance for 'Urgent' class (if supported)
    top_features_path = os.path.join(model_dir, "top_features_urgent.txt")
    save_top_features_urgent(best_model, top_k=20, output_path=top_features_path)
    print(f"[INFO] Top urgent features (if available) saved to {top_features_path}")

    # 8. Save best model and its metrics (for assignment requirement)
    final_metrics = {
        "best_model": best_model_name,
        "f1_best_model": best_f1,
        "metrics_" + best_model_name: all_metrics[best_model_name],
    }

    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"[INFO] Final metrics saved to {metrics_path}")

    joblib.dump(best_model, model_path)
    print(f"[INFO] Best model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train urgent message detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
