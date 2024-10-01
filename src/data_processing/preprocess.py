import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Remove duplicates and handle missing values
    df = df.drop_duplicates().fillna(df.mean())

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Protocol', 'Service', 'Flag'])

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Split features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
