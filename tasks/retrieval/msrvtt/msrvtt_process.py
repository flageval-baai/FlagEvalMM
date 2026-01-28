import os
import zipfile
from huggingface_hub import hf_hub_download


# from dataloaders.rawframe_util import RawFrameExtractor
def process(cfg):
    download_and_extract_dataset(cfg.dataset_path, cfg.processed_dataset_path)


def download_and_extract_dataset(repo_id, extract_dir):
    """
    Download Hugging Face dataset and extract it to the specified path.

    Parameters:
        repo_id (str): The name of the dataset (e.g., "shiyili1111/MSR-VTT").
        extract_dir (str): The directory to download and extract the dataset.
    """
    # Ensure the extraction directory exists
    os.makedirs(extract_dir, exist_ok=True)

    # Download the dataset files using hf_hub_download
    try:
        # Download the ZIP file
        zip_file_path = hf_hub_download(
            repo_id=repo_id,
            filename="MSRVTT_Videos.zip",
            repo_type="dataset",
            local_dir=extract_dir,
        )
        print(f"ZIP file downloaded successfully to: {zip_file_path}")

        # Download the CSV file
        csv_file_path = hf_hub_download(
            repo_id=repo_id,
            filename="MSRVTT_JSFUSION_test.csv",
            repo_type="dataset",
            local_dir=extract_dir,
        )
        print(f"CSV file downloaded successfully to: {csv_file_path}")

    except Exception as e:
        print(f"Download failed: {e}")
        return  # If download fails, return immediately

    # Extract the ZIP file
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"ZIP file extracted successfully to: {extract_dir}")
    except zipfile.BadZipFile:
        print("Error: The file is not a valid ZIP file.")
    except FileNotFoundError:
        print(f"Error: The file '{zip_file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
