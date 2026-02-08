import argparse
import sys

from image_classification.inference import (
    DEFAULT_IMAGE_SIZE,
    load_model,
    predict_probability,
    probability_to_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick benchmark validation on two images")
    parser.add_argument("--indoor", required=True, help="Expected indoor image path")
    parser.add_argument("--outdoor", required=True, help="Expected outdoor image path")
    parser.add_argument(
        "--model-path",
        default="./training_1/saved_model",
        help="Path to saved model directory/file",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
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
    model = load_model(args.model_path)
    image_size = tuple(args.image_size)

    indoor_prob = predict_probability(model, args.indoor, image_size)
    outdoor_prob = predict_probability(model, args.outdoor, image_size)

    indoor_pred = probability_to_label(indoor_prob, args.threshold)
    outdoor_pred = probability_to_label(outdoor_prob, args.threshold)

    print(f"indoor_image_p_outdoor={indoor_prob:.4f} predicted={indoor_pred}")
    print(f"outdoor_image_p_outdoor={outdoor_prob:.4f} predicted={outdoor_pred}")

    passed = indoor_pred == "indoor" and outdoor_pred == "outdoor"
    if not passed:
        print("benchmark_result=FAIL")
        sys.exit(1)
    print("benchmark_result=PASS")


if __name__ == "__main__":
    main()
