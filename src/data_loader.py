import pandas as pd

def load_data(filepath):
    """Lädt Daten aus einer CSV-Datei."""
    return pd.read_csv(filepath)
