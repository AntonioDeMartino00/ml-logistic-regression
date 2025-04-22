import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df):
    sns.countplot(x='HeartDisease', data=df)
    plt.title('target distribution')
    plt.show()

def plot_age_distribution(df):
    sns.histplot(df['Age'], kde=True)
    plt.title('age distribution')
    plt.show()

def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('correlationmatrix')
    plt.show()

def print_target_class_distribution(df):
    print(df['HeartDisease'].value_counts())
    
def print_column_info(df):
    print("datatypes of the columns:")
    print(df.dtypes)
    print("\ncolnames:")
    print(df.columns)
    print("\ncollist:")
    print(df.columns.tolist())
