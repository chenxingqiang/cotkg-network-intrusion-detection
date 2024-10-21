import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove duplicates and handle missing values
    df = df.drop_duplicates().fillna(df.mean())
    
    # Identify categorical and numerical columns
    categorical_columns = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
    numerical_columns = [col for col in df.columns if col not in categorical_columns and col != 'label']
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # No need to encode categorical variables as they seem to be already in binary format
    
    # Split features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
