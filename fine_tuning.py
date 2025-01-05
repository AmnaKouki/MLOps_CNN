import os
import numpy as np
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to the dataset directories
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Data generators for training and testing
def get_data_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode="binary",
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode="binary",
    )
    return train_generator, test_generator

# Build the CNN model
def build_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Run fine-tuning experiments
def run_experiments():
    experiments = [
        {"epochs": 20, "batch_size": 8},
        {"epochs": 20, "batch_size": 16},
        {"epochs": 25, "batch_size": 16},
    ]

    for i, exp in enumerate(experiments):
        print(f"Running Experiment {i + 1}: {exp}")

        train_gen, test_gen = get_data_generators(batch_size=exp["batch_size"])

        model = build_model()

        with mlflow.start_run(run_name=f"Experiment {i + 1}"):
            # Log hyperparameters
            mlflow.log_param("epochs", exp["epochs"])
            mlflow.log_param("batch_size", exp["batch_size"])

            # Train the model
            history = model.fit(
                train_gen, validation_data=test_gen, epochs=exp["epochs"]
            )

            # Log metrics
            val_loss, val_accuracy = model.evaluate(test_gen)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_accuracy)

            # Save the model and log as an artifact
            model.save(f"model_exp_{i + 1}.h5")
            mlflow.log_artifact(f"model_exp_{i + 1}.h5")

            # Save and log accuracy/loss plots
            plot_metrics(history, f"metrics_exp_{i + 1}.png")
            mlflow.log_artifact(f"metrics_exp_{i + 1}.png")

# Function to plot accuracy and loss
def plot_metrics(history, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow server URI
    mlflow.set_experiment("CNN Fine-Tuning Experiments")
    run_experiments()
