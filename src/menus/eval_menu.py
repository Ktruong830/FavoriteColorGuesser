import matplotlib.pyplot as plt

# Calculates and outputs the training error, validation error, and test error
def display_set_error(model, X_train, y_train, X_val, y_val, X_test, y_test):
    train_mse, train_mae = model.evaluate(X_train, y_train)
    val_mse, val_mae = model.evaluate(X_val, y_val)
    test_mse, test_mae = model.evaluate(X_test, y_test)

    print()
    print(f"Training MSE: {train_mse:.4f}   |   MAE: {train_mae:.4f}")
    print(f"Validation MSE: {val_mse:.4f} |   MAE: {val_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}       |   MAE: {test_mae:.4f}")
    print()

# Displays a line graph of the MSE of the training and validation sets in relation to the epochs
def plot_loss_curves(history):
    # Creates the graph
    plt.figure(figsize=(8, 5))
    # Creates title and labels
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    # Creates the training loss line
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='blue')
    # Creates the validation loss line
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='orange')
    plt.legend() # Creates the legend
    plt.tight_layout()
    plt.show()

