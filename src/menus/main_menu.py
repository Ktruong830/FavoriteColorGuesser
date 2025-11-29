from src.question_types import *
from src.menus.eval_menu import *
from src.menus.test_menu import *

# Prompts the user to fill out the survey to predict their favorite color
def user_survey(model, survey):
    take_survey = True
    while take_survey:
        os.system("cls")
        print("Favorite Color Guessing Survey")
        name = input("Enter your name: ").strip() # Gets the user's name
        user_answers_encoded = [] # Will store the user's encoded answers
        # Goes through the questions in the survey
        for question, options in survey.items():
            answer = multiple_choice_question(question, options) # Gets the user's answer to a question
            user_single_answer_encoded = len(options) * [0] # Stores one hot encoded data for
                                                            # a single question for the user
            user_answer_index = options.index(answer) # Find the index of the answer
            user_single_answer_encoded[user_answer_index] = 1 # Set the corresponding index to 1
            user_answers_encoded.extend(user_single_answer_encoded)

        # Converts to NumPy array
        user_answers_encoded = np.array([user_answers_encoded])
        # Makes a prediction
        prediction = model.predict(user_answers_encoded)

        # Gets the predicted circular values (sin/cos hue in [-1,1], saturation/value in [0,1])
        predicted_rgb = denormalize_model_output(prediction[0], "RGB")
        predicted_hsv = denormalize_model_output(prediction[0], "HSV")

        # Display the prediction results
        print("\nPredicted Favorite Color:")
        print(f"Predicted RGB: ({predicted_rgb[0]}, {predicted_rgb[1]}, {predicted_rgb[2]})")
        print(
            f"Predicted HSV: ({predicted_hsv[0]:.0f}°, {predicted_hsv[1]:.0f}%, {predicted_hsv[2]:.0f}%)\n")

        # Provides visual for the predicted color
        plt.imshow([[predicted_rgb]])
        plt.axis('off')
        plt.title(f"{name}'s Predicted Favorite Color")
        plt.tight_layout()
        plt.show()

        # Asks the user if they want to retake the survey
        while True:
            take_survey = input("Take survey again? (yes/no): ").strip().lower()
            if take_survey == "yes":
                break
            elif take_survey == "no":
                take_survey = False
                break
            else:
                continue

# Displays a general summary of the model
def display_model_overview(model, X_train, y_train, y_val, y_test, survey):
    print()
    print("===== Model Architecture =====") # Displays the model architecture
    model.summary()
    print()
    print("===== Survey Information =====") # Displays information about the survey
    print("Number of Survey Questions:", len(survey))
    print("Total Encoded Input Features:", len(X_train[0]))
    print()
    print("===== Dataset Sizes =====") # Displays information about the sizes of the datasets
    print("Training Set Size:", y_train.shape[0])
    print("Validation Set Size:", y_val.shape[0])
    print("Test Set Size:", y_test.shape[0])
    print()

# Allows the user to access a menu to evaluates the model
def evaluate_model(model, history, X_train, y_train, X_val, y_val, X_test, y_test):
    # Repeats the menu select until the user exits
    while True:
        print("\nModel Evaluation Submenu")
        print("1) Display Set Errors")
        print("2) Training & Validation Loss vs. Epochs Line Graph")
        print("3) Exit Submenu")
        try:
            menu_select = int(input("Your selection: ").strip())
        except ValueError:
            continue
        match menu_select:
            case 1:
                display_set_error(model, X_train, y_train, X_val, y_val, X_test, y_test)
            case 2:
                plot_loss_curves(history)
            case 3:
                break
            case _:
                continue

# Allows the user to access a menu to analyze the test set
def analyze_test(model, test_names, X_test, y_test):
    # Makes predictions on the test set
    prediction = model.predict(X_test)
    actual = y_test
    # Denormalizes and converts to NumPy arrays
    predicted_hsv = np.array([denormalize_model_output(p, "HSV") for p in prediction])
    actual_hsv = np.array([denormalize_model_output(a, "HSV") for a in actual])
    # Repeats the menu select until the user exits
    while True:
        print("\nTest Set Analysis Submenu")
        print("1) Display Test Set Individual Prediction & Actual RGB/HSV")
        print("2) Predicted vs. Actual Hues Polar Graph")
        print("3) Predicted vs. Actual Saturation & Value Scatter Plot")
        print("4) Exit Submenu")
        try:
            menu_select = int(input("Your selection: ").strip())
        except ValueError:
            continue
        match menu_select:
            case 1:
                show_test_details(predicted_hsv, actual_hsv, test_names)
            case 2:
                plot_hue_predictions(predicted_hsv, actual_hsv, test_names)
            case 3:
                plot_sat_val_predictions(predicted_hsv, actual_hsv, test_names)
            case 4:
                break
            case _:
                continue