import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Dataset configuration
ddd_train_path = "/kaggle/input/driver-drowsiness-dataset-ddd/Driver Drowsiness Dataset (DDD)"
mrl_train_path = "/kaggle/input/mrl-eye-dataset/data/train"
mrl_val_path = "/kaggle/input/mrl-eye-dataset/data/val"
mrl_test_path = "/kaggle/input/mrl-eye-dataset/data/test"

batch_size = 32
img_size = (224, 224)

# Verify directory structure
def verify_directory_structure(path):
    subdirs = [f.name for f in os.scandir(path) if f.is_dir()]
    if len(subdirs) != 2:
        raise ValueError(f"Directory {path} must contain exactly 2 subdirectories.")
    print(f"{path} contains valid subdirectories: {subdirs}")

verify_directory_structure(ddd_train_path)
verify_directory_structure(mrl_train_path)

# Load datasets
def load_dataset(path):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

ddd_train_ds = load_dataset(ddd_train_path)
mrl_train_ds = load_dataset(mrl_train_path)
val_ds = load_dataset(mrl_val_path).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
test_ds = load_dataset(mrl_test_path).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# Combine training datasets
train_ds = ddd_train_ds.concatenate(mrl_train_ds)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Normalize + augment
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return data_augmentation(image), label

train_ds = train_ds.map(preprocess)

# Model definition
def create_model():
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

# Phase 1: Train with frozen layers
history_initial = model.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks)

# Report training accuracy
initial_acc = history_initial.history['accuracy'][-1]
initial_val_acc = history_initial.history['val_accuracy'][-1]
print(f"Initial Training Accuracy: {initial_acc:.3f}, Validation Accuracy: {initial_val_acc:.3f}")

# Phase 2: Fine-tune entire model
for layer in model.layers:
    if isinstance(layer, layers.Conv2D):
        layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])

history_fine = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks)

# Report fine-tuned accuracy
fine_acc = history_fine.history['accuracy'][-1]
fine_val_acc = history_fine.history['val_accuracy'][-1]
print(f"Fine-tuned Training Accuracy: {fine_acc:.3f}, Validation Accuracy: {fine_val_acc:.3f}")

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Final Test Accuracy: {test_acc:.3f}")

# Save model
model.save("final_model.keras")
model.save("final_model.h5")
print("Model saved in 'final_model.keras' and 'final_model.h5'")
