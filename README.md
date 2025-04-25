## **CameraBench: Towards Understanding Camera Motions in Any Video**  

| [🏠**Home Page**](https://linzhiqiu.github.io/papers/camerabench/) | [&#129303;**HuggingFace**](https://huggingface.co/datasets/syCen/CameraBench) | [**📖Paper**](https://arxiv.org/abs/2504.15376) |

## Taxonomy of Camera Motion Primitives

![Demo GIF](./images/3.gif)
Here we demonstrate our taxonomy includes three reference frames (object-, ground-, and camera-centric) and defines key motion types, including translation (e.g., upward), rotation (e.g., roll clockwise), intrinsic changes (e.g., zoom-in), circular motion (e.g., arcing), steadiness (e.g., shaky), and tracking shots (e.g., side-tracking).


## CameraBench

We introduce CameraBench, a large-scale dataset with over 150K binary labels and captions over ~3,000 videos spanning diverse types, genres, POVs, capturing devices, and post-production effects (e.g., nature, films, games, 2D/3D, real/synthetic, GoPro, drone shot, etc.). We showcase example annotations below:
![Demo GIF](./images/4.gif)

These annotations allow us to evaluate and improve the performance of SfMs and VLMs on a wide range of tasks (video-text retrieval, video captioning, video QA, etc.) that require both geometric and semantic understanding of camera motion. We show example video QA tasks below:
![Demo GIF](./images/6.1.gif)
![Demo GIF](./images/6.2.gif)
![Demo GIF](./images/6.3.long.gif)


# SfMs vs. VLMs on CameraBench
![Demo GIF](./images/5.gif)
![Demo GIF](./images/sfm_vs_vlm.jpg)

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
