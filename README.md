## **CameraBench: Towards Understanding Camera Motions in Any Video**  

| [🏠**Home Page**](https://linzhiqiu.github.io/papers/camerabench/) | [&#129303;**HuggingFace**](https://huggingface.co/datasets/syCen/CameraBench) | [**📖Paper**](https://arxiv.org/abs/2504.15376) |

## Taxonomy of Camera Motion Primitives

![Demo GIF](./images/3.gif)
Here we demonstrate our taxonomy includes three reference frames (object-, ground-, and camera-centric) and defines key motion types, including translation (e.g., upward), rotation (e.g., roll clockwise), intrinsic changes (e.g., zoom-in), circular motion (e.g., arcing), steadiness (e.g., shaky), and tracking shots (e.g., side-tracking).


## CameraBench

We introduce CameraBench, a large-scale dataset with over 150K binary labels and captions over ~3,000 videos spanning diverse types, genres, POVs, capturing devices, and post-production effects (e.g., nature, films, games, 2D/3D, real/synthetic, GoPro, drone shot, etc.). We showcase example annotations below:
![Demo GIF](./images/4.gif)

### News

### **🚀 Quick Start**
```python
from video_data import VideoData
from camera_motion_data import camera_motion_params_demo
import json

# Create a VideoData object
video_sample = VideoData()

# 🔹 Initializing cam_motion
# You can initialize cam_motion with a dictionary of parameters or a CameraMotionData instance
# However, you should never create a CameraMotionData instance directly without using its create() function.

video_sample.cam_motion = camera_motion_params_demo  # Correct way to set

# 🔹 Displaying camera_motion_params_demo dictionary
print("camera_motion_params_demo:")
print(json.dumps(camera_motion_params_demo, indent=4))

# 🔹 Trying to access an uninitialized attribute (this will raise an error)
print(f"If you try to access cam_setup before setting it, it will raise an Error.")
try:
    print(video_sample.cam_setup)
except AttributeError as e:
    print(f"AttributeError: {e}")
```

---

### **🔹 Rules for Initialization**
✅ You **must use** the `.create()` function for `CameraMotionData`, `CameraSetupData`, and `LightingSetupData`.  
✅ You **should not** create instances of these classes manually.  
✅ Uninitialized attributes will **raise an `AttributeError`** when accessed.  

---

### Downloading testing videos from huggingface
```python
# 🤗 Get the test data from Hugging Face

import os
import json
import argparse
import subprocess
from tqdm import tqdm

HF_PREFIX = "https://huggingface.co/datasets/syCen/CameraBench/resolve/main/videos"

def save_to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def download_hf_video(video_names, save_dir="hf_videos"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"📥 Downloading {len(video_names)} videos to {save_dir}")
    failed_videos = []
    successful_videos = []

    for video_name in tqdm(video_names, desc="Downloading videos", unit="video"):
        path = os.path.join(save_dir, video_name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            successful_videos.append(video_name)
            continue

        url = f"{HF_PREFIX}/{video_name}"
        result = subprocess.run(
            ["wget", url, "-O", path, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0:
            successful_videos.append(video_name)
        else:
            print(f"❌ Failed to download {video_name}")
            if os.path.exists(path):
                os.remove(path)
            failed_videos.append(video_name)

    print(f"\n✅ Successfully downloaded {len(successful_videos)} videos")
    print(f"❌ Failed to download {len(failed_videos)} videos")

    if failed_videos:
        failed_path = os.path.join(save_dir, "failed_videos.json")
        save_to_json(failed_videos, failed_path)
        print(f"💾 Saved failed video list to: {failed_path}")

    return successful_videos

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos from Hugging Face using video_name list from JSON.")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the JSON file containing video metadata with 'video_name' fields.")
    parser.add_argument("--save_dir", type=str, default="hf_videos",
                        help="Directory to save the downloaded videos.")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    video_names = [item["video_name"] for item in data if "video_name" in item]

    download_hf_video(video_names, save_dir=args.save_dir) 
```

---

# SfMs vs. VLMs on CameraBench
We highlight the following key findings:

- Recent learning-based SfM/SLAM methods like [MegaSAM](https://arxiv.org/abs/2412.04463) and [CuT3R](https://cut3r.github.io/) achieve superior performance across most motion primitives, significantly outperforming classic methods like COLMAP. Nonetheless, SfMs are still far from solving this task. We show failure cases of SfM methods below:
![Demo GIF](./images/5.gif)
  - *Left:* A `lead-tracking` shot where the camera moves backward as the subject walks forward. Due to unchanged subject framing and lack of distinct background textures, MegaSAM fails to detect camera translation and COLMAP crashes. 
  - *Right:* A `roll-clockwise` shot in a low-parallax scene where both MegaSAM and COLMAP fail to converge and output random trajectories with nonexistent motion.

- Although generative VLMs (evaluated using [VQAScore](https://linzhiqiu.github.io/papers/vqascore/)) are weaker than SfM/SLAM, they generally outperform discriminative VLMs that use CLIPScore/ITMScore. Furthermore, they are able to capture the **semantic primitives** that depend on scene content, while SfMs struggle to do so. Motivated by this, we apply supervised fine-tuning (SFT) to a generative VLM (Qwen2.5-VL) on a separately annotated training set of ~1400 videos. We show that simple SFT on small-scale (yet high-quality) data significantly boosts performance by 1-2x, making it match the SOTA MegaSAM in overall AP.
![Demo GIF](./images/sfm_vs_vlm.jpg)

## Visual Demonstrations

<table>
  <tr>
    <td><img src="VQA-Leaderboard.png" alt="VQA Leaderboard" width="400"></td>
    <td>
      <img src="8-1.gif" alt="Animation 1" width="400"><br>
      <img src="8-2.gif" alt="Animation 2" width="400"><br>
      <img src="8-3.gif" alt="Animation 3" width="400"><br>
      <b>Question</b>: Does the camera move forward during the video?
    </td>
  </tr>
</table>

## Citation

If you find this repository useful for your research, please use the following.
```
@article{lin2025towards,
  title={Towards Understanding Camera Motions in Any Video},
  author={Lin, Zhiqiu and Cen, Siyuan and Jiang, Daniel and Karhade, Jay and Wang, Hewei and Mitra, Chancharik and Ling, Tiffany and Huang, Yuhan and Liu, Sifan and Chen, Mingyu and Zawar, Rushikesh and Bai, Xue and Du, Yilun and Gan, Chuang and Ramanan, Deva},
  journal={arXiv preprint arXiv:2504.15376},
  year={2025},
}
```
