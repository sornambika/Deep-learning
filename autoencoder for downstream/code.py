import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
# Step 1: Create and Prepare Your Data
# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, # Number of samples
 n_features=20, # Number of features
 n_informative=15, # Number of informative features
 n_redundant=5, # Number of redundant features
 n_classes=3, # Number of classes
 random_state=42)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 2: Build the Autoencoder
# Define the autoencoder architecture
input_dim = X_train.shape[1] # Number of features
# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(64, activation='relu')(input_layer)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.Dense(16, activation='relu')(encoded)
# Latent space
latent_space = layers.Dense(8, activation='relu')(encoded)
# Decoder
decoded = layers.Dense(16, activation='relu')(latent_space)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(64, activation='relu')(decoded)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)
# Autoencoder model
autoencoder = models.Model(input_layer, output_layer)
# Compile the mode
autoencoder.compile(optimizer='adam', loss='mse')
# Summary of the model
autoencoder.summary()
# Step 3: Train the Autoencoder
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)
# Step 4: Extract Features Using the Encoder
# Extract the encoder part of the autoencoder
encoder = models.Model(input_layer, latent_space)
# Encode the training and test data
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)
# Step 5: Build and Train a Classifier
# Build a classifier using the encoded features
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the classifier
classifier.fit(X_train_encoded, y_train)
# Predict on the test set
y_pred = classifier.predict(X_test_encoded)
# Step 6: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
# Optional: Evaluate using original features for comparison
classifier_original = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_original.fit(X_train, y_train)
y_pred_original = classifier_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
report_original = classification_report(y_test, y_pred_original)
print(f"Accuracy with Original Features: {accuracy_original}")
print(f"Classification Report with Original Features:\n{report_original}")
