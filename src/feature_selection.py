import pandas as pd

def split_features_and_target(df, target_column):
    X = df.drop(target_column, axis=1) 
    y = df[target_column]  
    return X, y
