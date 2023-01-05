import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    """Train a collaborative filtering model on the training data.
    
    Parameters:
    X (pd.DataFrame): The training data, containing the derived features for each user.
    y (pd.Series): The training labels, containing the ratings for each user-product pair.
    
    Returns:
    RandomForestRegressor: The trained model.
    """
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Evaluate the performance of a collaborative filtering model on the test data.
    
    Parameters:
    model (RandomForestRegressor): The trained model.
    X (pd.DataFrame): The test data, containing the derived features for each user.
    y (pd.Series): The test labels, containing the ratings for each user-product pair.
    
    Returns:
    float: The mean absolute error of the model on the test data.
    """
    y_pred = model.predict(X)
    return mean_absolute_error(y, y_pred)

def train_and_evaluate(X, y):
    """Train and evaluate a collaborative filtering model on the given data.
    
    Parameters:
    X (pd.DataFrame): The training and test data, containing the derived features for each user.
    y (pd.Series): The training and test labels, containing the ratings for each user-product pair.
    
    Returns:
    float: The mean absolute error of the model on the test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train)
    return evaluate_model(model, X_test, y_test)
