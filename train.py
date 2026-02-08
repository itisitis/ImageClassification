import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train indoor/outdoor classifier")
    parser.add_argument(
        "--train-dir",
        default="./images",
        help="Dataset directory with one subfolder per class",
    )
    parser.add_argument(
        "--output-dir",
        default="./training_1",
        help="Directory to store checkpoints and saved model",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split in [0,1)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[380, 380],
        metavar=("HEIGHT", "WIDTH"),
        help="Training image size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_model(image_size: tuple[int, int], learning_rate: float) -> tf.keras.Model:
    base_model = EfficientNetB4(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    if not 0 <= args.validation_split < 1:
        raise ValueError("--validation-split must be in [0, 1)")

    image_size = tuple(args.image_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.keras"
    saved_model_path = output_dir / "saved_model.keras"

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        validation_split=args.validation_split,
    )

    train_generator = datagen.flow_from_directory(
        args.train_dir,
        target_size=image_size,
        batch_size=args.batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=args.seed,
    )

    val_generator = datagen.flow_from_directory(
        args.train_dir,
        target_size=image_size,
        batch_size=args.batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )

    model = build_model(image_size, args.learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(saved_model_path)
    print(f"saved_model={saved_model_path}")


if __name__ == "__main__":
    main()
