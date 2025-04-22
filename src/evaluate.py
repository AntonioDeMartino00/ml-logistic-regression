import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def display_model_coefficients(model, X_train):
    coefficients = pd.DataFrame(model.coef_.flatten(), X_train.columns, columns=['coefficient'])
    print(coefficients)
    return coefficients


def compare_actual_vs_predicted(y_test, y_pred):
    comparison = pd.DataFrame({'test_values': y_test, 'predict_values': y_pred})
    print(comparison)
    return comparison


def evaluate_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy


def evaluate_classification_metrics(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("confusion matrix:\n", conf_matrix)
    return conf_matrix


