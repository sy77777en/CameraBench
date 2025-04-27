import argparse
import os
import requests
from huggingface_hub import list_repo_files
from tqdm import tqdm

def download_videos(repo_id, folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 列出 repo 里的所有文件
    all_files = list_repo_files(repo_id, repo_type="dataset")
    video_files = [f for f in all_files if f.startswith(folder + "/") and f.endswith(".mp4")]

    print(f"Found {len(video_files)} video files in '{folder}'.")

    for file_path in tqdm(video_files, desc="Downloading videos"):
        filename = file_path.split("/")[-1]
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
        local_path = os.path.join(save_dir, filename)

        if os.path.exists(local_path):
            continue  # 如果已经存在就跳过

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_path, "wb") as f, tqdm(
                total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename, leave=False
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

    print("✅ All videos downloaded successfully!")

def main():
    parser = argparse.ArgumentParser(description="Download videos from a Hugging Face dataset folder.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save downloaded videos.")
    args = parser.parse_args()

    repo_id = "syCen/CameraBench"  # 固定你的repo
    folder = "videos"              # 固定你的子目录
    download_videos(repo_id, folder, args.save_dir)

if __name__ == "__main__":
    main()
