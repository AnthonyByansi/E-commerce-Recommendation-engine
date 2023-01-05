import pandas as pd

def read_ratings(path):
    """Read in the ratings data from a CSV file.
    
    Parameters:
    path (str): The file path to the ratings data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the ratings data.
    """
    return pd.read_csv(path)

def filter_ratings(df, min_rating=4):
    """Filter the ratings data to include only ratings that are greater than or equal to a specified minimum.
    
    Parameters:
    df (pd.DataFrame): The ratings DataFrame.
    min_rating (int): The minimum rating to include in the filtered data.
    
    Returns:
    pd.DataFrame: A filtered DataFrame containing only ratings that are greater than or equal to the specified minimum.
    """
    return df[df['rating'] >= min_rating]

def normalize_ratings(df):
    """Normalize the ratings data using min-max normalization.
    
    Parameters:
    df (pd.DataFrame): The ratings DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing the normalized ratings.
    """
    df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
    return df

def generate_derived_features(df):
    """Generate derived features for each user in the ratings data.
    
    Parameters:
    df (pd.DataFrame): The ratings DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing the derived features for each user.
    """
    features = pd.DataFrame()
    features['mean_rating'] = df.groupby('user_id')['rating'].mean()
    features['rating_variance'] = df.groupby('user_id')['rating'].var()
    features['rating_count'] = df.groupby('user_id')['rating'].count()
    return features
