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

if __name__ == '__main__':
    # Read in the preprocessed data
    X = pd.read_csv('data/processed/derived_features.csv')
    y = pd.read_csv('data/processed/ratings_normalized.csv')['rating']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    mae = evaluate_model(model, X_test, y_test)
    print(f'Mean Absolute Error: {mae:.3f}')
