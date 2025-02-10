# ControlEdit: A MultiModal Local Clothing Image Editing Method
<a href="https://arxiv.org/abs/2409.14720"><img src="https://img.shields.io/badge/Arxiv-2408.00653-B31B1B.svg"></a> <a href="https://huggingface.co/chengzhiyuan/ControlEdit"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://cd123-cd.github.io/ControlEdit/"><img src="https://img.shields.io/badge/Project-Website-orange"></a>
![Result Image](./web/banner.jpg)
🔥News
- [07/20/2024] Our [paper](https://bmvc2024.org/proceedings/723/) has been accepted by The British Machine Vision Association and Society for Pattern Recognition  (BMVC)!
  
**Abstract:**
Multimodal clothing image editing refers to the precise adjustment and modification of clothing images using data such as textual 
descriptions and visual images as control conditions, which effectively improves the work efficiency of designers and reduces the threshold 
for user design. In this paper, we propose a new image editing method-ControlEdit, which transfers clothing image editing to multimodal-
guided local inpainting of clothing images. We address the difficulty of collecting real image datasets by leveraging the self-supervised learning 
approach. Based on this learning approach, we extend the channels of the feature extraction network to ensure consistent clothing image 
style before and after editing, and we design an inverse latent loss function to achieve soft control over the content of non-edited 
areas. In addition, we use Blended Latent Diffusion as the sampling method to make the editing boundaries transition naturally and enforce
 consistency of non-edited area content. Extensive experiments demonstrate that ControlEdit surpasses baseline algorithms in both 
qualitative and quantitative evaluations.
## Environment & Pre-trained models

### Dependancies
```
conda env create -f environment.yaml
conda activate controledit
```
### Checkpoints
- Place a downloaded file as below:
    ```
      ControlEdit/
            └── models/
                  ├── controledit_sd1.5_v1.ckpt
                  └── v1-5-pruned.ckpt


### Download
- controledit_sd1.5_v1.ckpt: [HuggingFace](https://huggingface.co/chengzhiyuan/ControlEdit)
- v1-5-pruned.ckpt: [HuggingFace](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)

## Inference
```
python inference.py
```

## Citation
```BibTeX
@inproceedings{Cheng_2024_BMVC,
author    = {Di Cheng and Yingjie Shi and sun shixin and JiaFu Zhang and weijing wang and YULiu},
title     = {ControlEdit: A MultiModal Local Clothing Image Editing Method},
booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
publisher = {BMVA},
year      = {2024},
url       = {https://papers.bmvc2024.org/0723.pdf}
}
```
## Acknowledgement
This project is largely based on [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main). We appreciate their great work and contributions to the field.
