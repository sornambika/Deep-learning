import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os  # For directory operations
import matplotlib.pyplot as plt  # For plotting

# Define the path to save the CSV file
csv_directory = 'D:/DLtask/data'
csv_file_path = os.path.join(csv_directory, 'dummy_sensor_data.csv')

# Create the directory if it does not exist
if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)

# Create a DataFrame with dummy data
np.random.seed(42)  # For reproducibility
data = {
    'sensor_1': np.random.uniform(0.1, 1.0, 100),  # 100 random numbers between 0.1 and 1.0
    'sensor_2': np.random.uniform(0.1, 1.0, 100),  # 100 random numbers between 0.1 and 1.0
    'failure': np.random.randint(0, 2, 100)         # 100 random binary values (0 or 1)
}

df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv(csv_file_path, index=False)

# Load the dataset
data = pd.read_csv(csv_file_path)

# Split the data into features (X) and target label (y)
X = data.drop(columns=['failure'])  # Features (drop the label column)
y = data['failure']  # Labels (equipment failure)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential()

# Add input layer and first hidden layer
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Add more hidden layers with dropout for regularization
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))

# Output layer with sigmoid activation for binary classification
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Plotting the training and validation accuracy and loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
