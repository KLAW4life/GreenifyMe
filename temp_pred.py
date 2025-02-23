import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess the data
data = pd.read_csv('Daily Records - Miami.csv')
data['DATE'] = pd.to_datetime(data['DATE'])

# Select relevant columns for prediction
features = [
    'Max_Temperature (F)(HIGH)', 'Max_Temperature (F)(YEAR)', 
    'Max_Temperature (F)(LOW)', 'Min_Temperature (F)(HIGH)', 
    'Min_Temperature (F)(YEAR)', 'Min_Temperature (F)(LOW)', 
    'Precipitation (INCHES)', 'MONTH'
]
target = 'Max_Temperature (F)(HIGH)'  # Target variable to predict

# Extract features and target
feature_data = data[features]
target_data = data[[target]]

# Encode the 'MONTH' column using one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
month_encoded = encoder.fit_transform(feature_data[['MONTH']])
month_encoded_df = pd.DataFrame(month_encoded, columns=encoder.get_feature_names_out(['MONTH']))

# Drop the original 'MONTH' column and concatenate the encoded columns
feature_data = feature_data.drop(columns=['MONTH'])
feature_data = pd.concat([feature_data, month_encoded_df], axis=1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_feature_data = scaler.fit_transform(feature_data)
scaled_target_data = scaler.fit_transform(target_data)

# Combine features and target into a single array
scaled_data = np.hstack((scaled_feature_data, scaled_target_data))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # All columns except the last (target)
        y.append(data[i + seq_length, -1])     # Last column (target)
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])), kernel_regularizer='l2'),
#     Dropout(0.4),  # Increased dropout
#     LSTM(50, return_sequences=False, kernel_regularizer='l2'),
#     Dropout(0.4),  # Increased dropout
#     Dense(25, kernel_regularizer='l2'),
#     Dense(1)
# ])

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2]), kernel_regularizer='l2'),
    Dropout(0.4),  # Increased dropout
    LSTM(50, return_sequences=False, kernel_regularizer='l2'),
    Dropout(0.4),  # Increased dropout
    Dense(25, kernel_regularizer='l2'),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')

# Add learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=64,  # Increased batch size
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# # Generate future predictions
# def generate_future_predictions(model, last_sequence, future_steps, feature_data, scaler, encoder):
#     future_predictions = []
#     current_sequence = last_sequence

#     for i in range(future_steps):
#         next_prediction = model.predict(current_sequence.reshape(1, seq_length, X_train.shape[2]), verbose=0)

#         future_predictions.append(next_prediction[0, 0])
        
#         # Create a new data point for the next prediction
#         new_data_point = np.zeros((1, feature_data.shape[1]))
#         new_data_point[0, 0] = next_prediction[0, 0]  # Update Max_Temperature (F)(HIGH)
#         new_data_point[0, 1:] = feature_data.iloc[-1, 1:]  # Keep other features the same
        
#         # Append the new data point to the sequence
#         current_sequence = np.append(current_sequence[1:], [new_data_point], axis=0)

#     return np.array(future_predictions)
def generate_future_predictions(model, last_sequence, future_steps, feature_data, scaler, encoder):
    future_predictions = []
    current_sequence = last_sequence

    for i in range(future_steps):
        next_prediction = model.predict(current_sequence.reshape(1, seq_length, X_train.shape[2]), verbose=0)

        future_predictions.append(next_prediction[0, 0])
        
        # Create a new data point for the next prediction
        new_data_point = np.zeros((1, feature_data.shape[1]))
        new_data_point[0, 0] = next_prediction[0, 0]  # Update Max_Temperature (F)(HIGH)
        new_data_point[0, 1:] = feature_data.iloc[-1, 1:]  # Keep other features the same
        
        # Append the new data point to the sequence. Removed [] around new_data_point.
        current_sequence = np.append(current_sequence[1:], new_data_point, axis=0)

    return np.array(future_predictions)

# Get the last sequence from the training data
last_sequence = X_train[-1]

# Define the number of future steps (1 year = 365 days)
future_steps = 365

# Generate future predictions (scaled)
future_predictions_scaled = generate_future_predictions(model, last_sequence, future_steps, feature_data, scaler, encoder)

# Inverse transform the predictions to the original scale
future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))[:, 0]


# Get the last date in the dataset
last_date = data['DATE'].iloc[-1]

# Create future dates for the next year, starting from the day after the last date
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'DATE': future_dates, 'PREDICTED_MAX_TEMP': future_predictions})

# Debug print: Show the final predictions
print("Final Predictions:")
print(future_df.head())  # Print the first few rows of the predictions
print("-" * 40)

# Plot the predicted temperatures
plt.figure(figsize=(15, 6))
plt.plot(future_df['DATE'], future_df['PREDICTED_MAX_TEMP'], label='Predicted Max Temperature', color='blue')
plt.xlabel('Date')
plt.ylabel('Max Temperature (°F)')
plt.title('Predicted Max Temperatures for the Next Year')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

model.save('Temp_model.h5')