import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, ReLU, BatchNormalization
import matplotlib.pyplot as plt

# Autoencoder model
def build_autoencoder(input_shape):
    # Encoder
    inputs = Input(shape=input_shape)

    # Encoding layers
    x = Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Decoder
    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Output layer: Reconstruct high-resolution image
    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = Model(inputs, outputs)
    return autoencoder

# Generate synthetic data
def generate_synthetic_data(num_images, image_shape):
    return np.random.rand(num_images, *image_shape)  # Random images with pixel values in range [0, 1]

# Set input shape (low-resolution 32x32x3 image)
input_shape = (32, 32, 3)

# Generate synthetic training and validation data
X_train = generate_synthetic_data(500, input_shape)  # 500 training images
X_val = generate_synthetic_data(100, input_shape)    # 100 validation images

# Build and compile the autoencoder
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_val, X_val))

# Visualize original and reconstructed images
def plot_original_and_reconstructed(autoencoder, images, num_images=5):
    reconstructed_images = autoencoder.predict(images[:num_images])
    
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Display original
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        
        # Display reconstruction
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i])
        plt.axis('off')      
    plt.show()
# Select a few images from the validation set to visualize
plot_original_and_reconstructed(autoencoder, X_val)
