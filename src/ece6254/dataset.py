import os
import kagglehub
import shutil

def download_and_save(download_path):
    os.makedirs(download_path, exist_ok=True)
    
    # Download dataset
    downloaded_path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    
    # Copy the downloaded file to raw directory
    raw_file_path = os.path.join(download_path, os.path.basename(downloaded_path))

    if os.path.isdir(downloaded_path):
        # If target directory already exists, remove it to avoid copytree error
        if os.path.exists(raw_file_path):
            shutil.rmtree(raw_file_path)

        shutil.copytree(downloaded_path, raw_file_path)
    else:
        shutil.copy(downloaded_path, raw_file_path)

    print(f"Raw data saved to {raw_file_path}")
