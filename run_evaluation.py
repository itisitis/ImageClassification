import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from image_classification.inference import (
    DEFAULT_IMAGE_SIZE,
    load_model,
    predict_batch,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate indoor/outdoor model")
    parser.add_argument("--class_1", required=True, help="Path to indoor folder")
    parser.add_argument("--class_2", required=True, help="Path to outdoor folder")
    parser.add_argument(
        "--out-path",
        default="./evaluation.csv",
        help="Output CSV path (default: ./evaluation.csv)",
    )
    parser.add_argument(
        "--model-path",
        default="./training_1/saved_model",
        help="Path to a saved Keras model directory/file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for outdoor class",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=list(DEFAULT_IMAGE_SIZE),
        metavar=("HEIGHT", "WIDTH"),
        help="Input image size used by the model",
    )
    return parser.parse_args()


def list_images(folder: str) -> list[Path]:
    base = Path(folder)
    images = [p for p in base.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    return sorted(images)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    indoor = list_images(args.class_1)
    outdoor = list_images(args.class_2)
    if not indoor or not outdoor:
        raise ValueError("Both class folders must contain at least one image")

    y_true = [0] * len(indoor) + [1] * len(outdoor)
    image_paths = indoor + outdoor

    model = load_model(args.model_path)
    probs = predict_batch(model, image_paths, tuple(args.image_size))
    y_pred = [1 if p >= args.threshold else 0 for p in probs]

    report = classification_report(y_true, y_pred, target_names=["indoor", "outdoor"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df["accuracy"] = accuracy_score(y_true, y_pred)

    output_path = Path(args.out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)

    cm = confusion_matrix(y_true, y_pred)
    print(f"accuracy={accuracy_score(y_true, y_pred):.4f}")
    print("confusion_matrix=[[tn, fp], [fn, tp]]")
    print(cm.tolist())
    print(f"saved_report={output_path}")


if __name__ == "__main__":
    main()
