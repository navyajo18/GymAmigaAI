# THIS WILL CREATE FAKE DATA IN A CSV FILE
# 100 PEOPLE with varying cycle lengths from 21-35 days
# Each person has A USER ID 1-100 to better organize the csv file

import random
from faker import Faker
import csv
import os

# Initialize Faker
fake = Faker()

# List of Exercises
exercises = ['Walking', 'Yoga', 'Sports', 'Running',
             'Legs & Glutes', 'Arms', 'Back & Shoulders', 'Abs & Waist']

# Function to GENERATE random exercise data input, not predicted, the INPUT


def generate_exercise_data(cycle_day):
    # RANDOM for exercise name and intensity
    exercise_name = random.choice(exercises)
    exercise_intensity = random.randint(1, 10)
    # VARIED for pain level, everyone has diff pain levels, min/max avoids >10 or <0
    pain_level = min(max(exercise_intensity + random.randint(-2, 2), 0), 10)
    return exercise_name, exercise_intensity, pain_level

# Function to PREDICT the next day's exercise based on the current data


def predict_exercise(cycle_day, exercise_name, exercise_intensity, pain_level, past_data):
    if pain_level > 7:  # for extreme cases, change the EXERCISE
        # pick an exercise that isn't the one it's currently on
        next_exercise = random.choice(
            [ex for ex in exercises if ex != exercise_name])
        # and lower the INTENSITY
        next_intensity = max(exercise_intensity - random.randint(1, 3), 1)
    else:
        next_exercise = exercise_name
        # 0, 2 and not -2, 2 because intensity should increase because pain is less
        next_intensity = min(exercise_intensity + random.randint(0, 2), 10)

    # dictionary, for cycle day 1 --> exercise, intensity, pain level
    past_data[cycle_day] = {'exercise_name': exercise_name,
                            'exercise_intensity': exercise_intensity, 'pain_level': pain_level}

    return next_exercise, next_intensity

# Assign cycle lengths and user IDs

# MAIN METHOD
def generate_variety(num_users=1000): # Generates the full data set for each user and puts it in data array
    data = []
    for user_id in range(1, num_users + 1):
        cycle_length = random.randint(21, 35)  # random cycle length 21-35
        past_data = {}
        for cycle_day in range(1, cycle_length + 1):
            #for each day of the cycle, call generate exercise data function to make input data
            exercise_name, exercise_intensity, pain_level = generate_exercise_data(
                cycle_day) 
            #for each day of the cycle, call predict_exercise to make prediction data
            predicted_exercise, predicted_intensity = predict_exercise(
                cycle_day, exercise_name, exercise_intensity, pain_level, past_data
            )
            #add all of this info to data array 
            data.append([user_id, cycle_length, cycle_day, exercise_name,
                        exercise_intensity, pain_level, predicted_exercise, predicted_intensity])

    # Define the path relative to the script's location
    output_directory = 'model'

    # Save CSV inside the model folder within the environment
    output_file_path = os.path.join(output_directory, 'fake_data.csv')
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User ID', 'Cycle Length', 'Cycle Day', 'Exercise Name',
                         'Exercise Intensity', 'Pain Level', 'Pred. Exercise', 'Pred. Intensity'])
        writer.writerows(data)

# standard to execute the main method is if __name__ == "__main__":
if __name__ == "__main__":
    generate_variety(num_users=1000) # calls main method 1000 times 