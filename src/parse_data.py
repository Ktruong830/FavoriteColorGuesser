from sklearn.model_selection import train_test_split
import csv
import numpy as np

from color_conversion import *

# Extracts the profiles from a csv file and puts them in a dictionary
def parse_profiles(csv_reader, survey, profiles):
    for row in csv_reader:  # Iterates through the people's information
        if not any(cell.strip() for cell in row):
            continue  # Skips empty rows
        name = row[0].strip()  # Gets the respondent's name
        color = row[1]  # Gets the respondent's favorite color
        # Converts the color to a list
        color = [int(value.strip()) for value in color.split(',') if value.strip()]
        # Checks that there are exactly 3 RGB values between 0-255
        if len(color) != 3 or not all(0 <= value <= 255 for value in color):
            raise ValueError(f"{name} has invalid RGB values {color}. Must have 3 values between 0–255.")

        # Gets the respondent's answers
        answers = [answer.strip() for answer in row[2:]]
        # Checks that the respondent has 1 answer for every corresponding question
        if len(answers) != len(survey):
            raise ValueError(f"{name} has {len(answers)} answers, but there are {len(survey)} questions.")
        # Checks that all the respondent's answers are valid
        for question_index, (question, options) in enumerate(survey.items()):
            answer = answers[question_index]
            if answer not in options:
                raise ValueError(f"{name} gave invalid answer '{answer}' for question '{question}'.")
        # Adds the respondent's information to profiles
        profiles[name] = (color, answers)
    return profiles

# Prepares X_data, which will contain the features
def prepare_X(survey, profiles):
    X_data = []
    for name in profiles:
        respondent_answers_encoded = [] # Will store one-hot encoded data for a respondent
        respondent_answers_list = profiles[name][1] # Gets this respondent's list of answers
        for question_index, question in enumerate(list(survey.keys())): # Iterates over questions
            options_list = survey[question] # Gets options for this question
            respondent_single_answer_encoded = [0] * len(options_list) # Will store one-hot encoded data for the
                                                                       # respondent's answer to this question
            answer = respondent_answers_list[question_index]
            answer_index = options_list.index(answer)
            respondent_single_answer_encoded[answer_index] = 1
            respondent_answers_encoded.extend(respondent_single_answer_encoded) # Adds encoded answer
        X_data.append(respondent_answers_encoded) # Adds person's encoded features
    X_data = np.array(X_data) # Converts X_data to a NumPy array
    return X_data

# Prepares y_data, which will contain the targets
def prepare_y(profiles):
    y_data = []
    for name in profiles:
        rgb = profiles[name][0] # Original RGB values (0–255)
        hsv = rgb_to_hsv(rgb) # Converts to HSV in degrees/percent
        circular_hsv = hsv_to_circular(hsv) # Converts to circular format
        sin_h, cos_h, s, v = circular_hsv
        y_data.append([sin_h, cos_h, s / 100, v / 100]) # Adds person's normalized targets
                                                                # sin/cos hue in [-1,1], saturation/value in [0,1]
    y_data = np.array(y_data) # Converts y_data to a NumPy array
    return y_data

# Extracts information from base and test CSV files and creates dictionaries and datasets from it
def parse_data_files(data_table_filename, test_table_filename):
    # Format: survey["question"] = options[]
    survey = dict() # Stores the questions and their options
    # Format: prior_profiles["name"] = ([R,G,B], answers[])
    prior_profiles = dict() # Stores information of prior respondents
    # Reads the data table
    with open(data_table_filename, "r") as file:
        csv_reader = csv.reader(file)
        try:
            # Gets the first row, which has headers and questions
            question_line = next(csv_reader)
            # Gets the second row, which has the options to the questions
            options_line = next(csv_reader)
        except StopIteration: # Ends the program if the header or options row is missing
            raise ValueError("Data Table is missing header or options row.")
        if len(question_line) < 3 or len(options_line) < 3:
            raise ValueError("Data Table must have at least 3 columns (Name, Color, Questions/Options...)")

        # Parses the survey
        for col in range(2, len(question_line)): # Iterates over the questions
            question = question_line[col] # Gets a question
            options = options_line[col].strip("[]") # Gets the options
            # Converts options to a list
            options = [option.strip() for option in options.split(',') if option.strip()]
            if not options:
                raise ValueError(f"No options found for question '{question}'.")
            # Adds the question and corresponding options to survey
            survey[question] = options

        # Parses the prior respondent profiles
        prior_profiles = parse_profiles(csv_reader, survey, prior_profiles)

    # Prepares X_data and y_data, which will be NumPy arrays
    X_data = prepare_X(survey, prior_profiles)
    y_data = prepare_y(prior_profiles)

    # Splits the data in ColorDataTable into the training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Format: test_profiles["name"] = ([R,G,B], answers[])
    test_profiles = dict() # Stores information of test respondents
    # Reads the test table
    with open(test_table_filename, "r") as file:
        csv_reader = csv.reader(file)
        # Parses the test respondent profiles
        test_profiles = parse_profiles(csv_reader, survey, test_profiles)

    # Prepares X_test and y_test, which will be NumPy arrays
    X_test = prepare_X(survey, test_profiles)
    y_test = prepare_y(test_profiles)

    return survey, prior_profiles, test_profiles, X_train, X_val, X_test, y_train, y_val, y_test