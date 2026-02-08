import argparse

from image_classification.inference import (
    DEFAULT_IMAGE_SIZE,
    load_model,
    predict_probability,
    probability_to_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indoor/outdoor image classifier")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument(
        "--model-path",
        default="./training_1/saved_model",
        help="Path to a saved Keras model directory/file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for outdoor class (default: 0.5)",
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


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    model = load_model(args.model_path)
    probability = predict_probability(model, args.input, tuple(args.image_size))
    label = probability_to_label(probability, args.threshold)

    print(f"class={label}")
    print(f"p_outdoor={probability:.4f}")
    print(f"threshold={args.threshold:.2f}")


if __name__ == "__main__":
    main()
