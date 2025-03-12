import subprocess
import zipfile
import os

def download_and_extract_dataset(repo_id, cache_dir, extract_dir):
    """
    Download Hugging Face dataset and extract it to the specified path.

    Parameters:
        repo_id (str): The name of the dataset (e.g., "shiyili1111/MSR-VTT").
        cache_dir (str): The directory to cache the downloaded file.
        extract_dir (str): The directory to extract the dataset.
    """
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Download the dataset
    command = (
        f"export HF_ENDPOINT=https://hf-mirror.com && "
        f"huggingface-cli download {repo_id} --repo-type dataset --local-dir {cache_dir}"
    )
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Download successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Download failed!")
        print(e.stderr)
        return  # If download fails, return immediately

    # Locate the downloaded ZIP file
    zip_file_path = os.path.join(cache_dir, "MSRVTT_Videos.zip")  # Assume the ZIP file is named MSRVTT_Videos.zip
    if not os.path.exists(zip_file_path):
        print(f"Error: ZIP file '{zip_file_path}' not found.")
        return

    # Ensure the extraction directory exists
    os.makedirs(extract_dir, exist_ok=True)

    # Extract the ZIP file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"ZIP file extracted successfully to: {extract_dir}")
    except zipfile.BadZipFile:
        print("Error: The file is not a valid ZIP file.")
    except FileNotFoundError:
        print(f"Error: The file '{zip_file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")