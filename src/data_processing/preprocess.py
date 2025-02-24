import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def validate_required_columns(df):
    """Validate that required columns are present in the dataframe"""
    # Update required columns based on your actual data
    required_columns = {
        'Flow Duration',  # or 'flow_duration'
        'Protocol',      # or 'Protocol Type'
        'Label'          # or 'label'
    }
    
    # Try different column name variations
    column_variations = {
        'Flow Duration': ['Flow Duration', 'flow_duration', 'Duration'],
        'Protocol': ['Protocol', 'Protocol Type', 'protocol'],
        'Label': ['Label', 'label', 'class']
    }
    
    # Check for missing columns with variations
    missing_columns = set()
    for req_col, variations in column_variations.items():
        if not any(var in df.columns for var in variations):
            missing_columns.add(req_col)
    
    return len(missing_columns) == 0, missing_columns

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42, test_mode=False):
    """Load and preprocess the CICIDS2017 dataset"""
    try:
        # Load data
        if test_mode:
            # First, peek at the data to get class distribution
            df_peek = pd.read_csv(data_path, usecols=['Label'])
            class_counts = df_peek['Label'].value_counts()
            
            # Calculate samples per class for balanced sampling
            samples_per_class = min(2000, min(class_counts))
            total_samples = samples_per_class * len(class_counts)
            
            # Read chunks to get balanced samples
            chunks = []
            for class_name in class_counts.index:
                class_df = pd.read_csv(
                    data_path,
                    nrows=samples_per_class * 2,  # Read extra to ensure we get enough after filtering
                    skiprows=lambda x: x != 0 and df_peek.iloc[x-1]['Label'] != class_name
                )
                class_df = class_df[class_df['Label'] == class_name].head(samples_per_class)
                chunks.append(class_df)
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"Sampled {len(df)} records with balanced classes")
        else:
            df = pd.read_csv(data_path)
        
        print(f"Loaded data shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Map variations of column names
        column_mapping = {
            'flow_duration': 'flow_duration',
            'flow duration': 'flow_duration',
            'duration': 'flow_duration',
            'protocol': 'protocol',
            'protocol_type': 'protocol',
            'label': 'label',
            'class': 'label'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Validate columns
        valid, missing_cols = validate_required_columns(df)
        if not valid:
            print(f"Missing required columns: {missing_cols}")
            return None
        
        # Handle missing values
        df = df.dropna()
        
        # Convert label to numeric if it's categorical
        if df['label'].dtype == 'object':
            # Keep track of label mapping
            label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
            df['label'] = df['label'].map(label_mapping)
            print("\nLabel mapping:")
            for label, idx in label_mapping.items():
                print(f"  {label}: {idx}")
        
        # Print class distribution
        print("\nClass distribution:")
        for label, count in df['label'].value_counts().items():
            original_label = [k for k, v in label_mapping.items() if v == label][0]
            print(f"  {original_label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Ensure we have multiple classes
        if len(df['label'].unique()) < 2:
            raise ValueError(f"Need at least 2 classes, got {len(df['label'].unique())}")
        
        # Split features and target
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Convert all features to numeric
        numeric_columns = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                    numeric_columns.append(col)
                except:
                    print(f"Warning: Dropping non-numeric column {col}")
                    X = X.drop(col, axis=1)
            else:
                numeric_columns.append(col)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[numeric_columns])
        X_scaled = pd.DataFrame(X_scaled, columns=numeric_columns, index=X.index)
        
        # Remove constant features
        selector = VarianceThreshold(threshold=0)  # Remove features with zero variance
        X_no_constant = selector.fit_transform(X_scaled)
        constant_features = X_scaled.columns[~selector.get_support()].tolist()
        print(f"\nRemoved {len(constant_features)} constant features:")
        print(constant_features)
        
        # Keep only non-constant features
        X_scaled = X_scaled.loc[:, selector.get_support()]
        
        print(f"\nFeatures after removing constants: {X_scaled.shape[1]}")
        print("Remaining features:", X_scaled.columns.tolist())
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(y.unique())}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    file_path = "data/raw/CICIDS2017.csv"
    try:
        print("\nRunning in test mode...")
        result = load_and_preprocess_data(
            file_path,
            test_mode=True,
            test_size=0.2,
            random_state=42
        )
        
        if result is not None:
            X_train, X_test, y_train, y_test = result
            print("\nData loading successful!")
            print(f"Training data shape: {X_train.shape}")
            print(f"Testing data shape: {X_test.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Testing labels shape: {y_test.shape}")
            
            # Print class distribution in train and test sets
            print("\nTraining set class distribution:")
            print(y_train.value_counts(normalize=True))
            print("\nTest set class distribution:")
            print(y_test.value_counts(normalize=True))
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        import traceback
        traceback.print_exc()