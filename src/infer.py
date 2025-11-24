import argparse
import joblib
import os
import yaml


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Train the model first by running: python src/train.py"
        )
    return joblib.load(model_path)


def main(config_path: str, messages):
    cfg = load_config(config_path)
    model_path = cfg["paths"]["model_path"]

    model = load_model(model_path)

    print(f"[INFO] Loaded model from {model_path}")
    print("\n=== Predictions ===")
    preds = model.predict(messages)

    for msg, label in zip(messages, preds):
        print(f"[{label}] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer urgency of messages")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "messages",
        nargs="*",
        help="Messages to classify. Example: python src/infer.py 'Server down' 'Let's meet'",
    )

    args = parser.parse_args()

    if not args.messages:
        # default examples if none provided
        default_msgs = [
            "Server is down for all users, need urgent fix asap",
            "Let's catch up for lunch tomorrow",
        ]
        main(args.config, default_msgs)
    else:
        main(args.config, args.messages)
