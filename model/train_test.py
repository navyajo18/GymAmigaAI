from keras import layers  # not sure if im calling keras right
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import CSV file

data = pd.read_csv('model/fake_data.csv')

# Encode categorical variables (e.g., Exercise Name)
label_encoder = LabelEncoder()
data['Exercise Name'] = label_encoder.fit_transform(data['Exercise Name'])
data['Pred. Exercise'] = label_encoder.fit_transform(data['Pred. Exercise'])

# Save the classes to a file
np.save('model/classes.npy', label_encoder.classes_)


# Normalize = Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1
scaler = StandardScaler()  # Call function StandardScaler()
data[['Exercise Intensity', 'Pain Level']] = scaler.fit_transform(
    data[['Exercise Intensity', 'Pain Level']])


# Save the scaler's mean and scale to files
np.save('model/scaler_mean.npy', scaler.mean_)
np.save('model/scaler_scale.npy', scaler.scale_)


# Split the data into x and y (inputs and outputs)
X = data[['Cycle Day', 'Exercise Name', 'Exercise Intensity', 'Pain Level']]
y = data[['Pred. Exercise', 'Pred. Intensity']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([  # sequential model = linear stack of layers
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # layer 1: fully connected, 64 neurons -- each neuron connects to all neurons in prev. layer
    layers.Dense(32, activation='relu'),
    # layer 2: fully connected, 32 neurons, takes input from first layer
    # just 2 neurons, output layer for Pred. Exercise and Pred. Intensity, we want RAW direct output values so no activation function
    layers.Dense(2)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# optimizer = 'adam' adjusts learning rate during training
# loss = 'mse' = how far is prediction from real value?
# mae = measures the error by averaging differences between predicted and actual

# TRAIN THE MODEL
history = model.fit(X_train, y_train, epochs=150,
                    validation_split=0.2, batch_size=32)

# Evaluating the Model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# EPOCHS Explanation.

# After each epoch, the model is being trained. The model makes predictions, calculates error, and adjusts weights using an optimizer (Adam)

# Now, it's time for the ...

# TESTING STEP: Making the prediction for NEW UNSEEN DATA using pandas dataframe

# Create new input DataFrame with the same structure as the training data

new_input = pd.DataFrame({
    'Cycle Day': [15],
    'Exercise Name': [label_encoder.transform(['Sports'])[0]],
    'Exercise Intensity': [4],
    'Pain Level': [2]
})

# Make sure the column order matches the order used during fitting
new_input_scaled = pd.DataFrame(scaler.transform(new_input[['Exercise Intensity', 'Pain Level']]),
                                columns=['Exercise Intensity', 'Pain Level'])

# Add the non-scaled features back to the DataFrame
new_input_scaled['Cycle Day'] = new_input['Cycle Day'].values
new_input_scaled['Exercise Name'] = new_input['Exercise Name'].values

# Reorder the columns to match the order expected by the model
new_input_scaled = new_input_scaled[[
    'Cycle Day', 'Exercise Name', 'Exercise Intensity', 'Pain Level']]

# Predict output
predicted_output = model.predict(new_input_scaled)
predicted_exercise = label_encoder.inverse_transform(
    [int(predicted_output[0][0])])
predicted_intensity = round(predicted_output[0][1])

# Output the predictions
print(f"Predicted Exercise: {predicted_exercise}")
print(f"Predicted Intensity: {predicted_intensity}")


# SAVE MODEL
model.save('model/trained_model.h5')
print("Model saved successfully.")
