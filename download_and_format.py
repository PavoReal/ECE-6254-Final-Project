import os
import kagglehub
import shutil

def download_and_save_raw():
    # Define directory for raw data
    raw_dir = 'raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download dataset using kagglehub
    downloaded_path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    
    # Copy the downloaded file to raw directory
    raw_file_path = os.path.join(raw_dir, os.path.basename(downloaded_path))
    if os.path.isdir(downloaded_path):
        # If target directory already exists, remove it to avoid copytree error
        if os.path.exists(raw_file_path):
            shutil.rmtree(raw_file_path)
        shutil.copytree(downloaded_path, raw_file_path)
    else:
        shutil.copy(downloaded_path, raw_file_path)
    print(f"Raw data saved to {raw_file_path}")

if __name__ == '__main__':
    download_and_save_raw() 