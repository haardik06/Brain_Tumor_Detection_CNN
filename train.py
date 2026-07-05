# ==========================================================
# Brain Tumor Detection using CNN
# Author : Hardik Verma
# ==========================================================

# ==========================
# Import Required Libraries
# ==========================

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# ==========================================================
# Project Configuration
# ==========================================================

# Dataset Paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Output Folders
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Create folders if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Image Configuration
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 4

print("=" * 60)
print("Brain Tumor Detection using CNN")
print("=" * 60)

# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading Training Dataset...\n")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.20,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

print("\nLoading Validation Dataset...\n")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

print("\nLoading Testing Dataset...\n")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================================
# Class Names
# ==========================================================

class_names = train_dataset.class_names

print("\nDetected Classes")

for index, name in enumerate(class_names):
    print(f"{index} --> {name}")

print()

# ==========================================================
# Optimize Dataset Performance
# ==========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# ==========================================================
# Data Augmentation
# ==========================================================

data_augmentation = keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.10),

    layers.RandomZoom(0.10),

])

print("\nDataset Loaded Successfully")

# ==========================================================
# Normalize Pixel Values
# ==========================================================

normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_dataset = test_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# ==========================================================
# Build CNN Model
# ==========================================================

print("\nBuilding CNN Model...\n")

model = keras.Sequential([

    # ------------------------------
    # Data Augmentation
    # ------------------------------
    data_augmentation,

    # ------------------------------
    # First Convolution Block
    # ------------------------------
    layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        padding="same",
        input_shape=(224,224,3)
    ),

    layers.MaxPooling2D(pool_size=(2,2)),


    # ------------------------------
    # Second Convolution Block
    # ------------------------------
    layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2,2)),


    # ------------------------------
    # Third Convolution Block
    # ------------------------------
    layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2,2)),


    # ------------------------------
    # Fourth Convolution Block
    # ------------------------------
    layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2,2)),


    # ------------------------------
    # Flatten Feature Maps
    # ------------------------------
    layers.Flatten(),

    # ------------------------------
    # Fully Connected Layer
    # ------------------------------
    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    # ------------------------------
    # Output Layer
    # ------------------------------
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])

# ==========================================================
# Display Model Summary
# ==========================================================

print(model.summary())

# ==========================================================
# Compile Model
# ==========================================================

print("\nCompiling Model...\n")

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("Model Compiled Successfully")

# ==========================================================
# Train Model
# ==========================================================

print("\nTraining Started...\n")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS

)

print("\nTraining Completed Successfully")

# ==========================================================
# Save Trained Model
# ==========================================================

print("\nSaving Model...\n")

MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")

model.save(MODEL_PATH)

print(f"Model saved successfully at:\n{MODEL_PATH}")

# ==========================================================
# Plot Training Accuracy
# ==========================================================

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    accuracy,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracy,
    marker="s",
    linewidth=2,
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_graph = os.path.join(
    RESULTS_DIR,
    "accuracy_graph.png"
)

plt.savefig(accuracy_graph)

plt.close()

print("Accuracy graph saved successfully.")

# ==========================================================
# Plot Training Loss
# ==========================================================

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8,6))

plt.plot(
    epochs_range,
    loss,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    marker="s",
    linewidth=2,
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_graph = os.path.join(
    RESULTS_DIR,
    "loss_graph.png"
)

plt.savefig(loss_graph)

plt.close()

print("Loss graph saved successfully.")

# ==========================================================
# Evaluate Model
# ==========================================================

print("\nEvaluating Model on Test Dataset...\n")

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("Final Test Results")
print("==============================")

print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ==========================================================
# Training Completed
# ==========================================================

print("\n===========================================")
print("Brain Tumor CNN Training Completed")
print("===========================================")

print("\nGenerated Files")

print("-------------------------------------------")

print("✔ Trained Model")

print("models/brain_tumor_cnn.keras")

print()

print("✔ Accuracy Graph")

print("results/accuracy_graph.png")

print()

print("✔ Loss Graph")

print("results/loss_graph.png")

print()

print("Project Finished Successfully")