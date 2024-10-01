import os
import requests
import zipfile
import hashlib
import pandas as pd
from tqdm import tqdm


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()


def check_md5(filename, md5_hash):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest() == md5_hash


def download_cicids2017():
    base_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/"
    zip_file = "MachineLearningCVE.zip"
    md5_file = "MachineLearningCSV.md5"

    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    # Download zip file
    print(f"Downloading {zip_file}...")
    download_file(base_url + "MachineLearningCSV.zip",
                  os.path.join('data/raw', zip_file))

    # Download md5 file
    print(f"Downloading {md5_file}...")
    download_file(base_url + md5_file, os.path.join('data/raw', md5_file))

    # Check integrity
    with open(os.path.join('data/raw', md5_file), 'r') as f:
        md5_hash = f.read().split()[0]

    if check_md5(os.path.join('data/raw', zip_file), md5_hash):
        print("File integrity check passed.")
    else:
        print("File integrity check failed. Please try downloading again.")
        return

    # Extract files
    print("Extracting files...")
    with zipfile.ZipFile(os.path.join('data/raw', zip_file), 'r') as zip_ref:
        zip_ref.extractall('data/raw')


def load_cicids2017():
    path = 'data/raw/MachineLearningCVE'
    all_files = [os.path.join(path, f)
                 for f in os.listdir(path) if f.endswith('.csv')]

    print("Loading CSV files...")
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, low_memory=False)
        df_list.append(df)

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    download_cicids2017()
    df = load_cicids2017()
    print(f"Loaded dataset with shape: {df.shape}")
