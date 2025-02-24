import os
import requests
import zipfile
import hashlib
import pandas as pd
from tqdm import tqdm
import sys
import time
import numpy as np
import joblib
import ssl
import urllib3
import subprocess

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_data():
    """Download and setup CICIDS2017 dataset"""
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Download dataset
    dataset_url = "https://www.unb.ca/cic/datasets/ids-2017.html"
    zip_path = "data/raw/CICIDS2017.zip"
    
    print("Downloading CICIDS2017 dataset...")
    download_file(dataset_url, zip_path)
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
    
    # Remove zip file
    os.remove(zip_path)
    print("Dataset setup complete!")

def download_cicids2017():
    """Download and extract the CICIDS2017 dataset from Kaggle"""
    try:
        import kaggle
    except ImportError:
        print("\nKaggle module not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

    # Create directories
    data_dir = os.path.join(os.getcwd(), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    dataset_dir = os.path.join(raw_dir, 'MachineLearningCVE')

    # Clean up any existing files
    if os.path.exists(dataset_dir):
        print("\nCleaning up existing files...")
        for file in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print("\nDownloading CICIDS2017 dataset from Kaggle...")
    
    # List of dataset IDs to try
    dataset_ids = [
        'galaxyh/cicids-2017',
        'cicdataset/cicids-2017',
        'dhoogla/cicids2017',
        'abubakar9999/cicids2017',
        'ranadeepsingh/cicids2017',
    ]
    
    for dataset_id in dataset_ids:
        try:
            print(f"\nTrying dataset: {dataset_id}")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_id,
                path=dataset_dir,
                unzip=True,
                quiet=False
            )
            
            # Verify the extracted files
            files = os.listdir(dataset_dir)
            if any(f.endswith(('.csv', '.parquet')) for f in files):
                print(f"\n\033[92m✓ Successfully downloaded from {dataset_id}\033[0m")
                print("\nFound files:")
                for f in files:
                    print(f"  - {f}")
                return True
                
        except Exception as e:
            print(f"\033[91mError with dataset {dataset_id}: {str(e)}\033[0m")
            continue
    
    print("\n\033[91mAll download attempts failed. Please:\033[0m")
    print("1. Visit https://www.kaggle.com/datasets/galaxyh/cicids-2017")
    print("2. Click 'Accept' on the dataset rules")
    print("3. Verify your Kaggle API credentials")
    print("4. Try running the script again")
    return False


def download_cicids2017_from_unb():
    """Download CICIDS2017 dataset from UNB website"""
    file_urls = [
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/MachineLearningCSV.zip",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Wednesday-workingHours.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]
    
    # Create directories
    data_dir = os.path.join(os.getcwd(), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    dataset_dir = os.path.join(raw_dir, 'MachineLearningCVE')
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    successful_downloads = 0
    
    for url in file_urls:
        filename = url.split('/')[-1]
        output_path = os.path.join(dataset_dir, filename)
        
        print(f"\nDownloading {filename}...")
        try:
            if download_file(url, output_path):
                if filename.endswith('.zip'):
                    print(f"Extracting {filename}...")
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    os.remove(output_path)
                successful_downloads += 1
                print(f"\033[92m✓ Successfully downloaded {filename}\033[0m")
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            continue
    
    # Verify files were downloaded
    files = os.listdir(dataset_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if csv_files:
        print(f"\n\033[92m✓ Downloaded {successful_downloads} files successfully\033[0m")
        print("\nFound files:")
        for f in csv_files:
            print(f"  - {f}")
        return True
    else:
        print("\n\033[91mNo CSV files found after download\033[0m")
        return False


def load_cicids2017():
    """Load the CICIDS2017 dataset into a pandas DataFrame"""
    extract_dir = os.path.join('data', 'raw', 'MachineLearningCVE')

    if not os.path.exists(extract_dir):
        print(f"Dataset directory not found: {extract_dir}")
        print("Please run download_cicids2017() first.")
        return None

    print("\nLoading data files...")
    try:
        # First try to find CSV files
        csv_files = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if f.endswith('.csv')
        ]
        
        # If no CSV files, try Parquet files
        parquet_files = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if f.endswith('.parquet')
        ]
        
        if not csv_files and not parquet_files:
            print("No CSV or Parquet files found in the dataset directory.")
            return None

        df_list = []
        
        # Try reading CSV files first
        if csv_files:
            print("Reading CSV files...")
            for filename in tqdm(csv_files, desc="Reading CSV files"):
                try:
                    df = pd.read_csv(filename, low_memory=False)
                    df_list.append(df)
                    print(f"Successfully loaded: {os.path.basename(filename)}")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
                    continue
        
        # If no CSV files were successfully read, try Parquet files
        if not df_list and parquet_files:
            print("\nInstalling pyarrow for Parquet support...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
            
            print("\nReading Parquet files...")
            for filename in tqdm(parquet_files, desc="Reading Parquet files"):
                try:
                    df = pd.read_parquet(filename, engine='pyarrow')
                    df_list.append(df)
                    print(f"Successfully loaded: {os.path.basename(filename)}")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
                    continue

        if not df_list:
            print("No valid files could be read.")
            return None

        print("\nCombining all dataframes...")
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"Final dataset shape: {combined_df.shape}")
        return combined_df

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def verify_dataset(df):
    """Verify the loaded dataset meets basic quality criteria"""
    verification_results = {
        'success': True,
        'messages': []
    }
    
    # Check for expected columns
    expected_columns = ['Label', 'Flow Duration', 'Total Fwd Packets']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        verification_results['success'] = False
        verification_results['messages'].append(f"Missing expected columns: {missing_cols}")
    
    # Check for minimum number of rows
    if len(df) < 1000:
        verification_results['success'] = False
        verification_results['messages'].append(f"Dataset too small: {len(df)} rows")
    
    # Check for excessive missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_pct[missing_pct > 5].index.tolist()
    if high_missing:
        verification_results['messages'].append(
            f"High missing values (>5%) in columns: {high_missing}")
    
    return verification_results


def prepare_data_for_training(csv_path=None, test_size=0.2, random_state=42):
    """
    Load and prepare the CICIDS2017 dataset for training models.
    
    Args:
        csv_path: Optional path to a specific CSV file. If None, loads from default directory
        test_size: Proportion of dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing processed data and metadata
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # If no specific CSV provided, load from default location
    if csv_path is None:
        df = load_cicids2017()
        if df is None:
            raise ValueError("Could not load dataset. Please ensure it's downloaded.")
    else:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    print("\nPreparing data for training...")
    
    # Remove rows with infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Removed {rows_removed:,} rows with missing/infinite values")

    # Separate features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Convert categorical features to numeric
    categorical_columns = X.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        X[column] = pd.to_numeric(X[column], errors='coerce')
    
    # Remove any remaining non-numeric columns
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X = X[numeric_columns]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Prepare metadata
    metadata = {
        'feature_names': X.columns.tolist(),
        'label_encoder': le,
        'scaler': scaler,
        'label_mapping': dict(zip(le.classes_, le.transform(le.classes_))),
        'n_features': X.shape[1],
        'n_classes': len(le.classes_),
        'class_distribution': dict(zip(le.classes_, np.bincount(y_encoded))),
    }
    
    # Print summary
    print("\nData preparation completed:")
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Testing samples: {len(X_test):,}")
    print(f"  - Features: {metadata['n_features']:,}")
    print(f"  - Classes: {metadata['n_classes']:,}")
    print("\nClass distribution:")
    for label, count in metadata['class_distribution'].items():
        print(f"  - {label}: {count:,} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': metadata
    }


def save_prepared_data(data_dict, output_dir='data/processed'):
    """
    Save the prepared data and metadata to disk.
    
    Args:
        data_dict: Dictionary containing the prepared data and metadata
        output_dir: Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), data_dict['X_train'])
    np.save(os.path.join(output_dir, 'X_test.npy'), data_dict['X_test'])
    np.save(os.path.join(output_dir, 'y_train.npy'), data_dict['y_train'])
    np.save(os.path.join(output_dir, 'y_test.npy'), data_dict['y_test'])
    
    # Save the metadata
    joblib.dump(data_dict['metadata'], os.path.join(output_dir, 'metadata.joblib'))
    
    print(f"\nSaved prepared data to {output_dir}/")


def load_prepared_data(data_dir='data/processed'):
    """
    Load the prepared data and metadata from disk.
    
    Args:
        data_dir: Directory containing the saved data
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    try:
        # Load the data arrays
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Load the metadata
        metadata = joblib.load(os.path.join(data_dir, 'metadata.joblib'))
        
        print("\nLoaded prepared data:")
        print(f"  - Training samples: {len(X_train):,}")
        print(f"  - Testing samples: {len(X_test):,}")
        print(f"  - Features: {metadata['n_features']:,}")
        print(f"  - Classes: {metadata['n_classes']:,}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error loading prepared data: {str(e)}")
        return None


if __name__ == "__main__":
    setup_data()
