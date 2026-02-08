# ImageClassification

Indoor vs outdoor image classification using TensorFlow/Keras and EfficientNet.

## What was improved

- Consolidated repeated inference logic into a reusable module: `image_classification/`.
- Refactored all scripts into robust CLIs with argument validation.
- Removed deprecated training APIs (`fit_generator`) and added validation + early stopping.
- Fixed evaluation correctness (`classification_report` now receives `y_true, y_pred` in proper order).
- Cleaned dependency definitions and added a practical `.gitignore`.

## Project structure

- `train.py`: model training
- `pred.py`: single-image inference
- `run_evaluation.py`: class-folder evaluation + CSV report
- `unit_test.py`: quick benchmark check on one indoor and one outdoor image
- `image_classification/inference.py`: shared prediction utilities

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

`train.py` expects:

```text
images/
  indoor/
    img1.jpg
    ...
  outdoor/
    img2.jpg
    ...
```

## Train

```bash
python train.py \
  --train-dir ./images \
  --output-dir ./training_1 \
  --epochs 10 \
  --batch-size 16
```

Model output is saved to `./training_1/saved_model.keras`.

## Inference

```bash
python pred.py \
  --input ./test_data/indoor/benchmark_in.jpg \
  --model-path ./training_1/saved_model.keras \
  --threshold 0.5
```

## Evaluation

```bash
python run_evaluation.py \
  --class_1 ./test_data/indoor \
  --class_2 ./test_data/outdoor \
  --model-path ./training_1/saved_model.keras \
  --out-path ./evaluation.csv
```

## Quick benchmark check

```bash
python unit_test.py \
  --indoor ./test_data/indoor/benchmark_in.jpg \
  --outdoor ./test_data/outdoor/benchmark_out.jpg \
  --model-path ./training_1/saved_model.keras
```

This command returns non-zero if predictions do not match expected classes.
