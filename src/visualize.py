import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df):
    sns.countplot(x='HeartDisease', data=df)
    plt.title('Verteilung der Zielvariable')
    plt.show()

def plot_age_distribution(df):
    sns.histplot(df['Age'], kde=True)
    plt.title('Alter der Patienten')
    plt.show()

def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Korrelationsmatrix')
    plt.show()

def print_target_class_distribution(df):
    print(df['HeartDisease'].value_counts())
    
def print_column_info(df):
    print("Datentypen der Spalten:")
    print(df.dtypes)
    print("\nSpaltennamen:")
    print(df.columns)
    print("\nSpaltenliste:")
    print(df.columns.tolist())
