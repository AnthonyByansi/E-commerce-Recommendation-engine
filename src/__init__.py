/*
The __init__.py file is a special file in Python that indicates that the directory it is located in should be treated as a package. It is typically used to include code that should be executed when the package is imported, such as to import certain modules or to define functions and variables.

*/

from . import preprocess
from . import train
from . import recommend

def read_ratings(path):
    return preprocess.read_ratings(path)

def filter_ratings(df, min_rating=4):
    return preprocess.filter_ratings(df, min_rating)

def normalize_ratings(df):
    return preprocess.normalize_ratings(df)

def generate_derived_features(df):
    return preprocess.generate_derived_features(df)

def train_and_evaluate(X, y):
    return train.train_and_evaluate(X, y)

def recommend(model, user_id, user_features, product_features, n=10):
    return recommend.recommend(model, user_id, user_features, product_features, n)
