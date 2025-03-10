import subprocess
import zipfile
import os

def download_and_extract_dataset(repo_id, cache_dir, extract_dir):
    """
    下载 Hugging Face 数据集并解压到指定路径。

    参数:
        repo_id (str): 数据集的名称（如 "shiyili1111/MSR-VTT"）。
        cache_dir (str): 下载文件的缓存目录。
        extract_dir (str): 解压缩的目标目录。
    """
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)

    # 下载数据集
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
        return  # 如果下载失败，直接返回

    # 找到下载的 ZIP 文件
    zip_file_path = os.path.join(cache_dir, "MSRVTT_Videos.zip")  # 假设 ZIP 文件名为 MSRVTT_Videos.zip
    if not os.path.exists(zip_file_path):
        print(f"Error: ZIP file '{zip_file_path}' not found.")
        return

    # 确保解压缩目标目录存在
    os.makedirs(extract_dir, exist_ok=True)

    # 解压缩 ZIP 文件
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

# 调用函数
def main():
    repo_id = "shiyili1111/MSR-VTT"  # 数据集名称
    cache_dir = "./msr-vtt1"         # 下载缓存目录
    extract_dir = "./msr-vtt1/extracted"  # 解压缩目标目录

    download_and_extract_dataset(repo_id, cache_dir, extract_dir)

# 运行主函数
if __name__ == "__main__":
    main()