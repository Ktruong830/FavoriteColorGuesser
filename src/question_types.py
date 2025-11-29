import os

# Displays and gets an answer for a multiple choice question
def multiple_choice_question(question, options):
    # Prints the question
    print(question)
    for i in range(len(options)): # Iterates through and prints the question options
        print(chr(97 + i) + ") " + options[i])
    # Get the user's answer
    while True:
        answer = input("Your choice: ").lower() # Stores the user's letter choice to the questions
        os.system("cls")
        # Makes sure the given letter is an available option
        if answer in [chr(97 + i) for i in range(len(options))]:
            break
    return options[ord(answer) - 97] # Returns the answer