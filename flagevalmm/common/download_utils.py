import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def download_file_with_progress(
    url, save_path=None, chunk_size=1024 * 1024, max_threads=4
):
    try:
        headers = requests.head(url, allow_redirects=True).headers
        file_size = int(headers.get("content-length", 0))
        file_name = os.path.basename(url.split("?")[0])

        if not file_name:
            content_disposition = headers.get("content-disposition", "")
            if "filename=" in content_disposition:
                file_name = content_disposition.split("filename=")[1].strip("\"'")
            else:
                file_name = f"downloaded_file_{int(time.time())}"

        if save_path is None:
            save_path = file_name
        elif os.path.isdir(save_path):
            save_path = os.path.join(save_path, file_name)
    except Exception as e:
        print(f"fail to get file information: {e}")
        return None

    if os.path.exists(save_path) and os.path.getsize(save_path) == file_size:
        print(f"file exist: {save_path}")
        return save_path

    temp_dir = f"{save_path}.temp"
    os.makedirs(temp_dir, exist_ok=True)

    def get_ranges(file_size, num_parts):
        part_size = file_size // num_parts
        ranges = []
        for i in range(num_parts):
            start = i * part_size
            end = (i + 1) * part_size - 1 if i < num_parts - 1 else file_size - 1
            ranges.append((start, end))
        return ranges

    ranges = get_ranges(file_size, max_threads)

    def download_part(part_num, start, end):
        part_file = os.path.join(temp_dir, f"part_{part_num}")
        headers = {"Range": f"bytes={start}-{end}"}

        while True:
            try:
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    with open(part_file, "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_chunks[part_num] += len(chunk)
                break
            except Exception as e:
                print(f"chunk {part_num} failed: {e}, retrying...")
                time.sleep(3)

    downloaded_chunks = [0] * max_threads
    last_update_time = time.time()
    last_downloaded = 0

    with tqdm(
        total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=file_name
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for i, (start, end) in enumerate(ranges):
                futures.append(executor.submit(download_part, i, start, end))

            while not all(f.done() for f in futures):
                time.sleep(0.1)
                total_downloaded = sum(downloaded_chunks)

                current_time = time.time()
                time_diff = current_time - last_update_time
                if time_diff >= 1.0:
                    speed = (total_downloaded - last_downloaded) / time_diff
                    pbar.set_postfix({"speed": f"{speed/1024/1024:.2f}MB/s"})
                    last_update_time = current_time
                    last_downloaded = total_downloaded

                pbar.update(total_downloaded - pbar.n)

        with open(save_path, "wb") as outfile:
            for i in range(max_threads):
                part_file = os.path.join(temp_dir, f"part_{i}")
                with open(part_file, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(part_file)

        os.rmdir(temp_dir)
        pbar.update(file_size - pbar.n)

    print(f"file saved at: {save_path}")
    return save_path


if __name__ == "__main__":
    url = "https://huggingface.co/datasets/fierytrees/UCF/resolve/main/UCF101.zip?download=true"
    save_path = "1.zip"

    download_file_with_progress(
        url=url, save_path=save_path, chunk_size=1024 * 1024, max_threads=4
    )
