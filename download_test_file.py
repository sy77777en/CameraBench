import argparse
import os
import requests

def download_single_file(repo_id, file_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"
    local_path = os.path.join(save_dir, os.path.basename(file_name))

    if os.path.exists(local_path):
        print(f"✅ '{file_name}' already exists. Skipping download.")
        return

    print(f"⬇️ Downloading {file_name} from {url}")
    response = requests.get(url)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Download completed: {local_path}")

def main():
    parser = argparse.ArgumentParser(description="Download a single file from a Hugging Face dataset repo.")
    parser.add_argument("--repo_id", type=str, default="syCen/CameraBench", help="Dataset repo ID on Hugging Face.")
    parser.add_argument("--file_name", type=str, default="test.jsonl", help="Path to file inside the repo (e.g., test.jsonl or metadata/test.jsonl).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the downloaded file.")
    
    args = parser.parse_args()
    download_single_file(args.repo_id, args.file_name, args.save_dir)

if __name__ == "__main__":
    main()
