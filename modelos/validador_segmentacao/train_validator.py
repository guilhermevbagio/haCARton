from pathlib import Path

import numpy as np
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
MODEL_PATH = (MODEL_DIR / "validador_segmentacao.keras").resolve()
BEST_WEIGHTS_PATH = (MODEL_DIR / "best_validador_segmentacao.weights.h5").resolve()
CLASSES_PATH = (MODEL_DIR / "classes.json").resolve()


def compute_class_weight_map(split_counts: dict[str, dict[str, int]], class_names: list[str]) -> dict[int, float]:
    train_counts = split_counts["train"]
    total = sum(train_counts.get(class_name, 0) for class_name in class_names)
    num_classes = max(1, len(class_names))
    weights: dict[int, float] = {}

    for idx, class_name in enumerate(class_names):
        count = train_counts.get(class_name, 0)
        if count <= 0:
            weights[idx] = 1.0
            continue
        weights[idx] = total / float(num_classes * count)

    return weights


def unfreeze_top_layers(base_model: tf.keras.Model, layers_to_unfreeze: int = 30) -> None:
    base_model.trainable = True
    if layers_to_unfreeze <= 0:
        for layer in base_model.layers:
            layer.trainable = False
        return

    cutoff = max(0, len(base_model.layers) - layers_to_unfreeze)
    for idx, layer in enumerate(base_model.layers):
        layer.trainable = idx >= cutoff
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
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
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="validador_segmentacao")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def main() -> None:
    print("Preparando dataset do validador...")
    source_counts = count_images_by_class(DATASET_SOURCE)
    print(f"Imagens encontradas em {DATASET_SOURCE}:")
    for class_name, count in source_counts.items():
        print(f"  - {class_name}: {count}")

    dataset_dir = split_dataset_if_needed(DATASET_SOURCE, DATASET_SPLIT, seed=RANDOM_SEED)
    split_counts = summarize_split_counts(dataset_dir)
    print(f"Dataset de treino/validação: {dataset_dir}")
    print("Divisão por split:")
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

    class_weights = compute_class_weight_map(split_counts, class_names)
    print("Pesos por classe no treino:")
    for idx, weight in class_weights.items():
        print(f"  - {class_names[idx]}: {weight:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, base_model = build_model(len(class_names))
    compile_model(model, learning_rate=1e-3)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(BEST_WEIGHTS_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    warmup_epochs = max(3, EPOCHS // 3)
    finetune_epochs = max(2, EPOCHS - warmup_epochs)

    print(f"Fase 1: treino da cabeca por {warmup_epochs} epocas.")
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print(f"Fase 2: fine-tuning das ultimas camadas por {finetune_epochs} epocas.")
    unfreeze_top_layers(base_model, layers_to_unfreeze=30)
    compile_model(model, learning_rate=1e-4)

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=history_warmup.epoch[-1] + 1,
        epochs=history_warmup.epoch[-1] + 1 + finetune_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    history = {
        key: history_warmup.history.get(key, []) + history_finetune.history.get(key, [])
        for key in set(history_warmup.history) | set(history_finetune.history)
    }

    if BEST_WEIGHTS_PATH.exists():
        model.load_weights(BEST_WEIGHTS_PATH)

    model.save(MODEL_PATH)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

    y_true_batches = []
    for _, labels in val_ds:
        y_true_batches.append(labels.numpy())
    y_true = np.concatenate(y_true_batches) if y_true_batches else np.array([], dtype=np.int32)

    y_pred_batches = []
    for images, _ in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred_batches.append(np.argmax(probs, axis=1))
    y_pred = np.concatenate(y_pred_batches) if y_pred_batches else np.array([], dtype=np.int32)
    class_mapping = {idx: name for idx, name in enumerate(class_names)}
    save_classes_json(class_mapping, CLASSES_PATH)

    print("Treinamento concluido.")
    print(f"Melhor modelo salvo em: {MODEL_PATH}")
    print(f"Mapeamento de classes salvo em: {CLASSES_PATH}")
    print(f"Loss final de validacao: {val_loss:.4f}")
    print(f"Accuracy final de validacao: {val_accuracy:.4f}")

    if y_true.size and y_pred.size:
        print("Resumo da validacao por classe:")
        for idx, class_name in enumerate(class_names):
            total = int(np.sum(y_true == idx))
            acertos = int(np.sum((y_true == idx) & (y_pred == idx)))
            recall = (acertos / total) if total else 0.0
            print(f"  - {class_name}: {acertos}/{total} corretos (recall={recall:.3f})")

    final_train_accuracy = history.get("accuracy", [None])[-1]
    final_train_loss = history.get("loss", [None])[-1]
    if final_train_accuracy is not None and final_train_loss is not None:
        print(f"Loss final de treino: {final_train_loss:.4f}")
        print(f"Accuracy final de treino: {final_train_accuracy:.4f}")


if __name__ == "__main__":
    main()
