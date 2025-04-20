import pandas as pd

def encode_categorical_features(df, categorical_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    return df_encoded
