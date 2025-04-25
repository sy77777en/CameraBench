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

# SfMs vs. VLMs on CameraBench
We highlight the following key findings:

- Recent learning-based SfM/SLAM methods like [MegaSAM](https://arxiv.org/abs/2412.04463) and [CuT3R](https://cut3r.github.io/) achieve superior performance across most motion primitives, significantly outperforming classic methods like COLMAP. Nonetheless, SfMs are still far from solving this task. We show failure cases of SfM methods below:
![Demo GIF](./images/5.gif)
  - *Left:* A `lead-tracking` shot where the camera moves backward as the subject walks forward. Due to unchanged subject framing and lack of distinct background textures, MegaSAM fails to detect camera translation and COLMAP crashes. 
  - *Right:* A `roll-clockwise` shot in a low-parallax scene where both MegaSAM and COLMAP fail to converge and output random trajectories with nonexistent motion.

- Although generative VLMs (evaluated using [VQAScore](https://linzhiqiu.github.io/papers/vqascore/)) are weaker than SfM/SLAM, they generally outperform discriminative VLMs that use CLIPScore/ITMScore. Furthermore, they are able to capture the **semantic primitives** that depend on scene content, while SfMs struggle to do so. Motivated by this, we apply supervised fine-tuning (SFT) to a generative VLM (Qwen2.5-VL) on a separately annotated training set of ~1400 videos. We show that simple SFT on small-scale (yet high-quality) data significantly boosts performance by 1-2x, making it match the SOTA MegaSAM in overall AP.
![Demo GIF](./images/sfm_vs_vlm.jpg)

# Evaluation Tables

Below are the evaluation results for **Video-Text Retrieval** and **Visual Question Answering (VQA)** tasks, formatted for clarity and readability. These tables compare various models on skill-based and caption-based tasks, highlighting the superior performance of our fine-tuned model, **Qwen2.5-VL-7B (SFT)**.

## Video-Text Retrieval Evaluation

This table compares **CLIPScore**, **ITMScore**, and **VQAScore** models on **skill-based** (evaluating 8 skills excluding Complex Description) and **caption-based** (evaluating Complex Description skill) video-text retrieval tasks. Metrics include **Text**, **Image**, and **Group** scores as defined in [NaturalBench, Winoground]. The results demonstrate that repurposing generative VLMs, particularly our **SFT model**, for discriminative scoring with VQAScore achieves state-of-the-art performance.

**Bold** indicates the best performance, and *italic* indicates the second-best.

| Model                     | **Skill-based Task** |         |         | **Caption-based Task** |         |         |
|                           | Text   | Image  | Group   | Text    | Image   | Group   |
| ------------------------- | ------ | ------ | ------- | ------- | ------- | ------- |
| Random Chance             | 25.0   | 25.0   | 16.6    | 25.0    | 25.0    | 16.6    |
| *CLIPScore*               | 21.6   | 5.8    | 3.5     | 44.0    | 26.7    | 19.8    |
| UMT-B16                   | 26.8   | 4.1    | 2.8     | 46.0    | 19.0    | 13.0    |
| UMT-L16                   | 23.7   | 4.4    | 2.6     | 39.5    | 17.3    | 11.1    |
| LanguageBind              | 24.0   | 9.7    | 6.2     | 53.6    | 39.6    | 33.2    |
| LanguageBindV1.5          | 24.1   | 8.3    | 5.4     | 55.9    | 38.7    | 33.0    |
| InternVideo2-S2           | 9.3    | 2.3    | 0.7     | 25.0    | 18.9    | 8.6     |
| *ITMScore*                | 17.6   | 9.5    | 4.3     | 42.7    | 37.2    | 25.3    |
| UMT-B16                   | 14.7   | 9.1    | 3.9     | 30.6    | 33.0    | 18.7    |
| UMT-L16                   | 19.9   | 10.7   | 5.0     | 45.2    | 37.0    | 26.2    |
| InternVideo2-S2           | 18.2   | 8.7    | 4.1     | 52.3    | 41.7    | 31.0    |
| *VQAScore*                | 28.3   | 39.7   | 20.5    | 54.2    | 53.0    | 39.0    |
| mPLUG-Owl3-7B             | 26.2   | 38.4   | 19.6    | 57.6    | 52.8    | 42.7    |
| LLaVA-OneVision-7B        | 24.3   | 39.7   | 18.8    | 56.4    | 53.0    | 40.9    |
| LLaVA-Video-7B            | 17.8   | 40.9   | 13.3    | 53.5    | 50.7    | 37.2    |
| InternVideo2-Chat-8B      | 21.4   | 18.0   | 8.0     | 41.2    | 26.3    | 16.1    |
| Tarsier-Recap-2           | 35.1   | 23.1   | 15.4    | 43.4    | 30.4    | 22.6    |
| InternLMXComposer-2.5-7B   | 14.3   | 33.0   | 9.8     | 40.4    | 54.2    | 29.5    |
| InternVL-2.5-8B           | 22.0   | 43.9   | 17.5    | 55.8    | 51.4    | 38.7    |
| InternVL-2.5-26B          | 22.1   | 45.1   | 18.7    | 57.4    | 54.2    | 39.1    |
| InternVL-3-8B             | 31.9   | <u>46.0</u> | 25.0    | 60.2    | 57.3    | 45.8    |
| InternVL-3-78B            | 35.7   | 44.6   | 26.8    | 63.4    | 60.5    | 48.2    |
| Qwen2.5-VL-7B             | 35.0   | 40.8   | 24.2    | 65.5    | 63.0    | 51.8    |
| Qwen2.5-VL-32B            | 41.4   | 42.7   | 29.5    | 65.6    | 67.7    | 53.0    |
| Qwen2.5-VL-72B            | <u>43.8</u> | 44.5   | <u>32.1</u> | <u>67.8</u> | <u>69.2</u> | <u>56.4</u> |
| GPT-4o                    | 38.3   | 42.4   | 25.8    | 39.9    | 40.3    | 31.6    |
| **Qwen2.5-VL-7B (SFT)**   | **60.3** | **79.9** | **57.7** | **89.2** | **87.0** | **82.0** |



## VQA Evaluation

This table reports **Accuracy (Acc)** and **Question Accuracy (Q-Acc)**, where Q-Acc awards a point only if *both* videos for a given question are answered correctly [NaturalBench]. The evaluation covers multiple skills, including Motion & Steadiness, Scene Dynamics, Motion Speed, Motion Direction, Confusable Motion, Has Motion, Shot Tracking, Only Motion, and Complex Description. **Bold** indicates the best performance, and *italic* indicates the second-best.

**Table 2: VQA evaluation.**  
We report both accuracy (**Acc**) and question-pair accuracy (**Q-Acc**) that scores a point only if *both* videos are answered correctly.

| Model                      | Motion & Steadiness     |         | Scene Dynamics      |         | Motion Speed        |         | Motion Direction    |         | Confusable Motion    |         | Has Motion           |         | Shot Tracking        |         | Only Motion          |         | Complex Desc.        |         | Overall              |         |
|                            | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    | Acc    | Q-Acc    |
| -------------------------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- | ------ | -------- |
| Random Chance              | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     | 50.0   | 25.0     |
| mPLUG-Owl3-7B              | 51.8   | 15.5     | <u>64.9</u> | 35.1     | 61.5   | 31.6     | 48.6   | 13.1     | 49.2   | 12.7     | 54.1   | 24.3     | 53.2   | 17.1     | 45.9   | 8.6      | 63.4   | 39.7     | 55.8   | 25.4     |
| LLaVA-Video-7B             | 53.5   | 12.8     | 66.1   | <u>36.2</u> | 57.2   | 22.4     | 52.1   | 17.8     | 49.9   | 5.4      | 54.9   | 13.9     | 59.9   | 29.2     | 51.3   | 2.9      | 68.0   | 41.8     | 58.8   | 24.1     |
| LLaVA-OneVision-7B         | 54.3   | 19.6     | 63.8   | 31.0     | **69.0** | <u>54.0</u> | 53.1   | 24.2     | <u>55.4</u> | 20.7     | 60.9   | 28.2     | <u>60.7</u> | 31.3     | 43.3   | 6.1      | 52.3   | 6.3      | 57.1   | 24.7     |
| InternVideo2-Chat-8B       | 52.4   | 13.7     | 64.4   | 31.6     | 51.7   | 5.2      | 50.2   | 2.9      | 49.7   | 13.8     | 52.2   | 5.5      | 48.5   | 2.3      | 50.9   | 4.3      | 50.6   | 1.3      | 51.3   | 5.3      |
| Tarsier-Recap-7B           | 51.8   | 12.3     | 62.8   | 29.2     | 50.5   | 4.8      | 49.8   | 2.5      | 49.0   | 12.5     | 51.5   | 5.0      | 47.8   | 2.0      | 50.2   | 3.8      | 49.8   | 1.0      | 50.6   | 4.8      |
| InternLMXComposer2.5-7B     | 52.8   | 12.8     | 57.8   | 19.5     | 56.6   | 17.2     | 49.6   | 1.7      | 53.3   | 14.8     | 53.2   | 9.9      | 49.1   | 11.6     | 51.2   | 2.4      | 48.4   | 7.8      | 51.7   | 9.3      |
| InternVL2.5-8B             | 54.4   | 14.9     | 59.8   | 23.0     | 57.5   | 31.6     | 51.3   | 12.8     | 49.7   | 0.0      | 58.1   | 22.5     | 55.2   | 14.1     | 50.0   | 0.0      | 50.0   | 0.0      | 54.5   | 16.7     |
| InternVL2.5-26B            | <u>56.2</u> | 17.3     | 63.5   | 26.4     | 60.8   | 35.2     | 53.8   | 15.6     | 51.2   | 14.5     | 60.3   | 25.8     | 58.4   | 18.9     | <u>52.5</u> | 2.4      | 53.6   | 3.8      | 57.2   | 19.8     |
| InternVL3-8B              | 54.4   | 14.9     | 59.8   | 23.0     | 57.5   | 31.6     | 51.3   | 12.8     | 49.7   | 0.0      | 58.1   | 22.5     | 55.2   | 14.1     | 50.0   | 0.0      | 50.0   | 0.0      | 54.5   | 16.7     |
| InternVL3-26B             | <u>56.2</u> | 17.3     | 63.5   | 26.4     | 60.8   | 35.2     | 53.8   | 15.6     | 51.2   | 14.5     | 60.3   | 25.8     | 58.4   | 18.9     | <u>52.5</u> | 2.4      | 53.6   | 3.8      | 57.2   | 19.8     |
| Qwen2.5-VL-7B              | 55.2   | 17.4     | 60.6   | 24.1     | 67.8   | 37.4     | 51.9   | 17.0     | 52.3   | 10.7     | 57.2   | 21.0     | 56.2   | 21.5     | 47.7   | 4.7      | 62.5   | 30.0     | 57.6   | 22.3     |
| Qwen2.5-VL-32B             | **65.5** | **39.9** | 59.8   | 25.3     | <u>69.3</u> | 46.0     | 55.1   | 26.0     | 51.7   | 22.6     | **66.0** | **41.0**   | 57.3   | 30.2     | 48.4   | 26.3     | 71.1   | 47.6     | 63.8   | 38.2     |
| Qwen2.5-VL-72B             | **67.2** | **42.1** | **60.5** | **26.8**   | **70.0** | **48.2**   | **56.8** | **28.3**   | **53.2** | <u>24.5</u>   | <u>67.3</u> | <u>42.6</u> | 59.0   | <u>32.4</u> | **50.3** | <u>28.8</u> | **72.8** | <u>50.1</u> | **65.2** | <u>40.7</u> |
| GPT-4o                    | 55.8   | <u>27.0</u> | 52.6   | 10.3     | 61.2   | 32.2     | <u>58.1</u> | <u>32.8</u> | 53.3   | 20.4     | 64.1   | 36.2     | 51.7   | 20.2     | 42.1   | 8.5      | 61.9   | 32.7     | 59.0   | 29.8     |
| Gemini-2-Flash            | 53.6   | 25.2     | 46.8   | 2.9      | 56.6   | 29.3     | 44.5   | 17.2     | 41.1   | 8.8      | 46.5   | 20.5     | 46.5   | 24.1     | 39.2   | 15.1     | 63.8   | 37.4     | 51.8   | 24.9     |
| Gemini-2.5-Pro            | 58.2   | 28.7     | 51.3   | 11.6     | 60.1   | 34.5     | 48.9   | 21.4     | 45.7   | 13.2     | 52.3   | 25.8     | 49.7   | 26.9     | 42.8   | 15.3     | <u>64.5</u> | 39.1     | 54.7   | 28.2     |
| **Qwen2.5-VL-7B (SFT)**   | **80.3** | **65.0** | **86.8** | **75.3**   | **87.4** | **76.4**   | **69.5** | **43.7**   | **59.2** | **35.3**   | **76.1** | **55.3**   | **85.5** | **72.7**   | **79.0** | **59.7**   | **83.5** | **68.5**   | **78.5** | **60.5**   |

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
