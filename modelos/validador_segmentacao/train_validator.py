from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

try:
    from .utils import (
        IMG_SIZE,
        RANDOM_SEED,
        build_datasets_from_directory,
        count_images_by_class,
        save_classes_json,
        split_dataset_if_needed,
        summarize_split_counts,
    )
except ImportError:
    from utils import (
        IMG_SIZE,
        RANDOM_SEED,
        build_datasets_from_directory,
        count_images_by_class,
        save_classes_json,
        split_dataset_if_needed,
        summarize_split_counts,
    )

BATCH_SIZE = 16
EPOCHS = 15
DATASET_SOURCE = Path("dataset_rotulado")
DATASET_SPLIT = Path("dataset_validador")
MODEL_DIR = Path("modelos/validador_segmentacao")
MODEL_PATH = MODEL_DIR / "validador_segmentacao.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"


def build_model(num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="validador_segmentacao")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    print("Preparando dataset do validador...")
    source_counts = count_images_by_class(DATASET_SOURCE)
    print(f"Imagens encontradas em {DATASET_SOURCE}:")
    for class_name, count in source_counts.items():
        print(f"  - {class_name}: {count}")

    dataset_dir = split_dataset_if_needed(DATASET_SOURCE, DATASET_SPLIT, seed=RANDOM_SEED)
    split_counts = summarize_split_counts(dataset_dir)
    print(f"Dataset de treino/valida??o: {dataset_dir}")
    print("Divis?o por split:")
    for split_name, counts in split_counts.items():
        print(f"  {split_name}:")
        for class_name, count in counts.items():
            print(f"    - {class_name}: {count}")

    train_ds, val_ds, class_names = build_datasets_from_directory(
        dataset_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    print(f"Classes detectadas: {class_names}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    class_mapping = {idx: name for idx, name in enumerate(class_names)}
    save_classes_json(class_mapping, CLASSES_PATH)

    print("Treinamento conclu?do.")
    print(f"Melhor modelo salvo em: {MODEL_PATH}")
    print(f"Mapeamento de classes salvo em: {CLASSES_PATH}")
    print(f"Loss final de valida??o: {val_loss:.4f}")
    print(f"Accuracy final de valida??o: {val_accuracy:.4f}")

    final_train_accuracy = history.history.get("accuracy", [None])[-1]
    final_train_loss = history.history.get("loss", [None])[-1]
    if final_train_accuracy is not None and final_train_loss is not None:
        print(f"Loss final de treino: {final_train_loss:.4f}")
        print(f"Accuracy final de treino: {final_train_accuracy:.4f}")


if __name__ == "__main__":
    main()
