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
        if isinstance(data_path, pd.DataFrame):
            # If data_path is already a DataFrame, use it directly
            df = data_path.copy()
            print(f"Using provided DataFrame with shape: {df.shape}")
        else:
            # Otherwise, treat it as a file path
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
                print(f"Loaded data from file with shape: {df.shape}")

        # Add a 'Label' column if it doesn't exist (for test data)
        if 'Label' not in df.columns and 'label' not in df.columns:
            df['Label'] = 0  # Default label for testing

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
            # Instead of returning None, add missing columns with default values
            for col in missing_cols:
                if col == 'Protocol':
                    df['protocol'] = 'TCP'  # Default protocol
                elif col == 'Flow Duration':
                    if 'duration' in df.columns:
                        df['flow_duration'] = df['duration']
                    else:
                        df['flow_duration'] = 0  # Default duration
                elif col == 'Label':
                    df['label'] = 0  # Default label

        # Handle missing values
        df = df.dropna()

        # Convert label to numeric if it's categorical
        if 'label' in df.columns and df['label'].dtype == 'object':
            # Keep track of label mapping
            label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
            df['label'] = df['label'].map(label_mapping)
            print("\nLabel mapping:")
            for label, idx in label_mapping.items():
                print(f"  {label}: {idx}")

        # Print class distribution
        if 'label' in df.columns:
            print("\nClass distribution:")
            for label, count in df['label'].value_counts().items():
                if df['label'].dtype == 'object' and 'label_mapping' in locals():
                    original_label = [k for k, v in label_mapping.items() if v == label][0]
                    print(f"  {original_label}: {count} ({count/len(df)*100:.1f}%)")
                else:
                    print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

            # Ensure we have multiple classes
            if len(df['label'].unique()) < 2:
                print("Warning: Only one class found in the data")

        # Return the preprocessed DataFrame
        return df

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
            df = result
            print("\nData loading successful!")
            print(f"Data shape: {df.shape}")

            # Print class distribution
            print("\nClass distribution:")
            print(df['label'].value_counts(normalize=True))
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        import traceback
        traceback.print_exc()