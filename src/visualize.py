import os
import yaml
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib


# Utility to load config

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


#  1. Model comparison bar charts
def plot_model_comparison(model_dir: str = "models"):
    """
    Reads model_comparison.csv (created by train.py) and
    draws bar charts for Accuracy and F1-score for each model.
    """
    comparison_path = os.path.join(model_dir, "model_comparison.csv")
    if not os.path.exists(comparison_path):
        print(f"[ERROR] model_comparison.csv not found at {comparison_path}")
        print("Run: python src/train.py first.")
        return

    df = pd.read_csv(comparison_path)

    # Expecting columns: model, accuracy, precision, recall, f1
    print("\n[INFO] Model comparison table:")
    print(df)

    # Accuracy bar chart
    plt.figure()
    plt.bar(df["model"], df["accuracy"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison - Accuracy")
    plt.tight_layout()
    plt.show()
    plt.close()

    # F1-score bar chart
    plt.figure()
    plt.bar(df["model"], df["f1"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1-score (Urgent)")
    plt.title("Model Comparison - F1-score for 'Urgent'")
    plt.tight_layout()
    plt.show()
    plt.close()


# 2. Confusion matrix for best model
def plot_confusion_matrix_for_best(config_path: str = "config.yaml"):
    """
    Rebuilds the same train/test split as in train.py,
    loads best_model.pkl, and plots confusion matrix.
    """
    cfg = load_config(config_path)

    data_path = cfg["data"]["path"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    test_size = cfg["train"]["test_size"]
    random_state = cfg["train"]["random_state"]

    model_path = cfg["paths"]["model_path"]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run train.py first.")
        return

    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Loading best model from {model_path}")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=["Normal", "Urgent"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Urgent"],
    )

    disp.plot(values_format="d")
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.show()
    plt.close()


# 3. Top urgent keywords bar chart
def plot_top_urgent_features(model_dir: str = "models"):
    """
    Reads top_features_urgent.txt (created by train.py) and
    plots a bar chart of the top-k features for 'Urgent' class.
    """
    top_feat_path = os.path.join(model_dir, "top_features_urgent.txt")
    if not os.path.exists(top_feat_path):
        print(f"[ERROR] top_features_urgent.txt not found at {top_feat_path}")
        print("Make sure train.py ran with save_top_features_urgent enabled.")
        return

    words = []
    weights = []

    with open(top_feat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # skip header lines or empty lines
            if not line or line.startswith("Top features"):
                continue
            # each line like: "word: 0.1234"
            if ":" in line:
                w, val = line.split(":", 1)
                w = w.strip()
                try:
                    val = float(val.strip())
                except ValueError:
                    continue
                words.append(w)
                weights.append(val)

    if not words:
        print("[WARN] No features parsed from top_features_urgent.txt")
        return

    # Plot as horizontal bar chart
    plt.figure()
    y_pos = range(len(words))[::-1]  # reverse to have top feature at top
    plt.barh(y_pos, weights[::-1])
    plt.yticks(y_pos, words[::-1])
    plt.xlabel("Weight")
    plt.title("Top TF-IDF Features for 'Urgent' Class (Best Model)")
    plt.tight_layout()
    plt.show()
    plt.close()


# Main
if __name__ == "__main__":
    print("=== Visualization Menu ===")
    print("1. Model comparison (Accuracy & F1)")
    print("2. Confusion matrix for best model")
    print("3. Top urgent keywords")
    choice = input("Enter choice (1/2/3, or 'all'): ").strip().lower()

    if choice == "1":
        plot_model_comparison()
    elif choice == "2":
        plot_confusion_matrix_for_best()
    elif choice == "3":
        plot_top_urgent_features()
    elif choice == "all":
        plot_model_comparison()
        plot_confusion_matrix_for_best()
        plot_top_urgent_features()
    else:
        print("Invalid choice.")