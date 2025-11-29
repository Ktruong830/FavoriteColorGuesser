# Kevin Truong
# 4/16/2025
# This neural network predicts the favorite color in HSV values

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from parse_data import parse_data_files
from src.menus.main_menu import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Builds the neural network model that will predict favorite color
def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.001)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(0.001)),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.001)),

        layers.Dense(4) # Output layer [hue sin, hue cos, saturation, value]
    ])
    return model

def main():
    # Prompt until a valid file is given
    while True:
        training_filename = input("Enter the filename of the training data: ")
        training_data_file = os.path.join(DATA_DIR, training_filename)
        if os.path.exists(training_data_file):
            break
        else:
            print(f"File '{training_filename}' not found. Please try again.")
    while True:
        test_filename = input("Enter the filename of the test data: ")
        test_data_file = os.path.join(DATA_DIR, test_filename)
        if os.path.exists(test_data_file):
            break
        else:
            print(f"File '{test_filename}' not found. Please try again.")

    # Loads the X and y training, validation, and test sets while getting the survey and profile dictionaries
    survey, prior_profiles, test_profiles, X_train, X_val, X_test, y_train, y_val, y_test = parse_data_files(training_data_file, test_data_file)
    test_names = list(test_profiles.keys()) # Gets the names of test respondents and puts it in a list

    # Creates the model
    input_shape = len(X_train[0]) # Gets the input shape
    model = build_model(input_shape)
    # Compiles the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # Trains the model, storing the history
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    # Repeats the menu select until the user exits
    while True:
        print("Menu")
        print("1) Take Survey")
        print("2) Display Model Overview")
        print("3) Evaluate Model Performance")
        print("4) Analyze Test Set")
        print("5) Exit")
        try:
            menu_select = int(input("Your selection: "))
        except ValueError:
            continue
        match menu_select:
            case 1:
                user_survey(model, survey)
            case 2:
                display_model_overview(model, X_train, y_train, y_val, y_test, survey)
            case 3:
                evaluate_model(model, history, X_train, y_train, X_val, y_val, X_test, y_test)
            case 4:
                analyze_test(model, test_names, X_test, y_test)
            case 5:
                break
            case _:
                continue
main()










