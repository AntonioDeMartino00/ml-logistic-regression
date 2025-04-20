import pandas as pd


def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def display_model_coefficients(model, X_train):
    coefficients = pd.DataFrame(model.coef_.flatten(), X_train.columns, columns=['Koeffizient'])
    print(coefficients)
    return coefficients
