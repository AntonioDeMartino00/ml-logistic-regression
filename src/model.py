from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, max_iter=5000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model
