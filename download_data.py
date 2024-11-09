import os
import requests
import zipfile
import hashlib
import pandas as pd
from tqdm import tqdm
import sys


def download_file(url, filename):
    """Download a file with progress bar and retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    progress_bar.update(size)
            progress_bar.close()
            return True
        except requests.RequestException as e:
            print(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            print("Retrying...")
    return False


def download_cicids2017():
    """Download and extract the CICIDS2017 dataset from the UNB mirror"""
    # Updated URLs for the UNB mirror
    base_url = "https://www.unb.ca/cic/datasets/ids-2017.html"
    file_urls = [
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

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download each file
    for url in file_urls:
        filename = url.split('/')[-1]
        output_path = os.path.join(dataset_dir, filename)

        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping...")
            continue

        print(f"\nDownloading {filename}...")
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False

    return True


def load_cicids2017():
    """Load the CICIDS2017 dataset into a pandas DataFrame"""
    extract_dir = os.path.join('data', 'raw', 'MachineLearningCVE')

    if not os.path.exists(extract_dir):
        print(f"Dataset directory not found: {extract_dir}")
        print("Please run download_cicids2017() first.")
        return None

    print("\nLoading CSV files...")
    try:
        all_files = [os.path.join(extract_dir, f)
                     for f in os.listdir(extract_dir)
                     if f.endswith('.csv')]

        if not all_files:
            print("No CSV files found in the dataset directory.")
            return None

        df_list = []
        for filename in tqdm(all_files, desc="Reading CSV files"):
            try:
                df = pd.read_csv(filename, low_memory=False)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
                continue

        if not df_list:
            print("No valid CSV files could be read.")
            return None

        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        return combined_df

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


if __name__ == "__main__":
    print("Starting CICIDS2017 dataset download and processing...")
    print("Note: This dataset is quite large and may take some time to download.")

    if not download_cicids2017():
        print("Download failed. Exiting...")
        sys.exit(1)

    df = load_cicids2017()
    if df is not None:
        print(f"\nLoaded dataset with shape: {df.shape}")

        # Print some basic statistics
        print("\nDataset Overview:")
        print(f"Number of features: {df.shape[1]}")
        if 'Label' in df.columns:
            print("\nLabel distribution:")
            print(df['Label'].value_counts())
    else:
        print("Failed to load dataset.")
        sys.exit(1)
