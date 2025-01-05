import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

# Paths to the dataset directories
TRAIN_DIR = "data/train"
TEST_DIR = 'data/test'

# Function to visualize sample images from the dataset
def plot_images(generator):
    x, y = next(generator)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        axes[i].imshow(x[i])
        axes[i].set_title(f"Label: {'Sea' if y[i] == 0 else 'Forest'}")
        axes[i].axis('off')
    plt.show()

def main():
    # Data Augmentation for training and preprocessing for testing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Loading data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    # Visualizing some training samples
    print("Visualizing sample training images...")
    plot_images(train_generator)

    # Building the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=20
    )

    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Save the model
    model.save("cnn_sea_forest_classifier.h5")
    print("Model saved as 'cnn_sea_forest_classifier.h5'")

    # Plot training history
    print("Visualizing training results...")
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
