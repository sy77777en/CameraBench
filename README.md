<p align="center">
  <img src="https://github.com/sy77777en/CameraBench/blob/main/images/CameraBench.png" width="600">
</p>

## 📷 **CameraBench: Towards Understanding Camera Motions in Any Video**  

### News
- **[2025/04/26]🔥** We open‑sourced our **fine‑tuned 7B model** and the public **test set**—1 000+ videos with expert labels & captions..
- **LLMs‑eval** integration is in progress—stay tuned!
- 32B & 72B checkpoints are on the way.

### 🌍Explore More
- [🤗**CameraBench Testset**](https://huggingface.co/datasets/syCen/CameraBench): Download the testset.
- [🚀**Lora Model**](): Access model checkpoints.
- [🏠**Home Page**](https://linzhiqiu.github.io/papers/camerabench/): Project Home Page.
- [📖**Paper**](https://arxiv.org/abs/2504.15376): Detailed information about CameraBench.
- [📈**Leaderboard**](): LeaderBoard.

---

## SfMs vs. VLMs on CameraBench
- Although generative VLMs (evaluated using [VQAScore](https://linzhiqiu.github.io/papers/vqascore/)) are weaker than SfM/SLAM, they generally outperform discriminative VLMs that use CLIPScore/ITMScore. Furthermore, they are able to capture the **semantic primitives** that depend on scene content, while SfMs struggle to do so. Motivated by this, we apply supervised fine-tuning (SFT) to a generative VLM (Qwen2.5-VL) on a separately annotated training set of ~1400 videos. We show that simple SFT on small-scale (yet high-quality) data significantly boosts performance by 1-2x, making it match the SOTA MegaSAM in overall AP.
![Demo GIF](./images/sfm_vs_vlm.jpg)

## VQA evaluation on VLMs

<table>
  <tr>
    <td>
    <div style="display: flex; flex-direction: column; gap: 1em;">
      <img src="./images/VQA-Leaderboard.png" width="440">
     </div>
    </td>
    <td>
      <div style="display: flex; flex-direction: column; gap: 1em;">
        <div>        
          <img src="./images/8-1.gif" width="405"><br>
          🤔: Does the camera track the subject from a side view? <br>
          🤖: ✅  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🙋: ✅
        </div>
        <div>
          <img src="./images/8-2.gif" width="405"><br>
          🤔: Does the camera only move down during the video? <br>
          🤖: ❌  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🙋: ✅
        </div>
        <div>
          <img src="./images/8-3.gif" width="405"><br>
          🤔: Does the camera move backward while zooming in? <br>
          🤖: ❌  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🙋: ✅
        </div>
      </div>
    </td>
  </tr>
</table>

## 🚀 Quick Start

### Download test videos
```python
python download_test_videos.py --save_dir ./your_target_folder
```

### Download corresponding captions and labels (part)
```python
python download_test_data.py --save_dir ./your_target_folder
```

### Download finetuned model
```python
```

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
