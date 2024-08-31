from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/trained_model.h5', compile=False)

# Load encoders and scaler
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('model/classes.npy', allow_pickle=True)

# Scale with old scaling method
scaler = StandardScaler()
scaler.mean_ = np.load('model/scaler_mean.npy')
scaler.scale_ = np.load('model/scaler_scale.npy')

# Load fake dataset for all users
fake_data = pd.read_csv('model/fake_data.csv')

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# ROUTE FOR ABOUT PAGE
@app.route('/about')
def about():
    return render_template('about.html')

# ROUTE FOR GET INSPIRED PAGE
@app.route('/get_inspired')
def get_inspired():
    return render_template('get_inspired.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    user_id = int(request.form['user_id'])
    cycle_day = int(request.form['cycle_day'])
    exercise_name = request.form['exercise_name']
    exercise_intensity = int(request.form['exercise_intensity'])
    pain_level = int(request.form['pain_level'])

    # Check if the User ID is valid
    if not (1 <= user_id <= 1000):
        return jsonify({'error': 'Invalid User ID. Please enter a User ID between 1 and 1000.'})

    # Filter the fake data for the user
    user_data = fake_data[fake_data['User ID'] == user_id]

    if user_data.empty:
        return jsonify({'error': 'No data found for this User ID.'})

    # Prepare the input data
    input_data = pd.DataFrame({
        'Cycle Day': [cycle_day],
        'Exercise Name': [label_encoder.transform([exercise_name])[0]],
        'Exercise Intensity': [exercise_intensity],
        'Pain Level': [pain_level]
    })

    # Scale the necessary features
    input_data[['Exercise Intensity', 'Pain Level']] = scaler.transform(
        input_data[['Exercise Intensity', 'Pain Level']])

    # Predict the output
    predicted_output = model.predict(input_data)
    predicted_exercise = label_encoder.inverse_transform(
        [int(predicted_output[0][0])])
    predicted_intensity = round(predicted_output[0][1])

    # Return the results as JSON
    return jsonify({
        'predicted_exercise': predicted_exercise[0],
        'predicted_intensity': predicted_intensity
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)