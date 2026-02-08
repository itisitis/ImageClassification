from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (380, 380)


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    return tf.keras.models.load_model(str(path))


def _load_image_tensor(image_path: str | Path, image_size: Tuple[int, int]) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = tf.keras.preprocessing.image.load_img(str(path), target_size=image_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    return preprocess_input(input_arr)


def predict_probability(
    model: tf.keras.Model,
    image_path: str | Path,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> float:
    """Return P(outdoor) for a single image."""
    image_tensor = _load_image_tensor(image_path, image_size)
    prediction = model.predict(image_tensor, verbose=0)
    return float(prediction[0][0])


def predict_batch(
    model: tf.keras.Model,
    image_paths: Iterable[str | Path],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> List[float]:
    return [predict_probability(model, image_path, image_size) for image_path in image_paths]


def probability_to_label(probability: float, threshold: float = 0.5) -> str:
    return "outdoor" if probability >= threshold else "indoor"

