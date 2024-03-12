# BrushNet

This repository contains the implementation of the paper "BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion"

Keywords: Image Inpainting, Diffusion Models, Image Generation

> [Xuan Ju](https://github.com/juxuan27)<sup>12</sup>, [Xian Liu](https://alvinliu0.github.io/)<sup>12</sup>, [Xintao Wang](https://xinntao.github.io/)<sup>1*</sup>, [Yuxuan Bian](https://scholar.google.com.hk/citations?user=HzemVzoAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Ying Shan](https://www.linkedin.com/in/YingShanProfile/)<sup>1</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2*</sup><br>
> <sup>1</sup>ARC Lab, Tencent PCG <sup>2</sup>The Chinese University of Hong Kong <sup>*</sup>Corresponding Author


<p align="center">
  <a href="https://tencentarc.github.io/BrushNet/">Project Page</a> |
  <a href="https://arxiv.org/abs/2403.06976">Arxiv</a> |
  <a href="https://forms.gle/9TgMZ8tm49UYsZ9s5">Data</a> |
  <a href="https://drive.google.com/file/d/1IkEBWcd2Fui2WHcckap4QFPcCI0gkHBh/view">Video</a> |
</p>



**📖 Table of Contents**




  - [🛠️ Method Overview](#️-method-overview)
  - [🚀 Getting Started](#-getting-started)
    - [Environment Requirement 🌍](#environment-requirement-)
    - [Data Download ⬇️](#data-download-️)
  - [🏃🏼 Running Scripts](#-running-scripts)
    - [Training 🤯](#training-)
    - [Inference 📜](#inference-)
  - [🤝🏼 Cite Us](#-cite-us)
  - [💖 Acknowledgement](#-acknowledgement)


## TODO

- [x] Release trainig and inference code
- [ ] Release checkpoint (sdv1.5)
- [ ] Release checkpoint (sdxl)
- [ ] Release evluation code
- [ ] Release gradio demo

## 🛠️ Method Overview

BrushNet is a diffusion-based text-guided image inpainting model that can be plug-and-play into any pre-trained diffusion model. Our architectural design incorporates two key insights: (1) dividing the masked image features and noisy latent reduces the model's learning load, and (2) leveraging dense per-pixel control over the entire pre-trained model enhances its suitability for image inpainting tasks. More analysis can be found in the main paper.

![](examples/brushnet/src/model.png)





## 🚀 Getting Started

### Environment Requirement 🌍

BrushNet has been implemented and tested on Pytorch 1.12.1 with python 3.9.

Clone the repo:

```
git clone https://github.com/TencentARC/BrushNet.git
```

We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/). For example:


```
conda create -n diffusers python=3.9 -y
conda activate diffusers
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, you can install diffusers (implemented in this repo) with:

```
pip install -e .
```

After that, you can install required packages thourgh:

```
cd examples/brushnet/
pip install -r requirements.txt
```

### Data Download ⬇️


**Dataset**

You can download the BrushData and BrushBench [here](https://forms.gle/9TgMZ8tm49UYsZ9s5) (as well as the EditBench we re-processed), which are used for training and testing the BrushNet. By downloading the data, you are agreeing to the terms and conditions of the license. The data structure should be like:

```
|-- data
    |-- BrushData
        |-- 00200.tar
        |-- 00201.tar
        |-- ...
    |-- BrushDench
        |-- images
        |-- mapping_file.json
    |-- EditBench
        |-- images
        |-- mapping_file.json
    |-- ckpt
        |-- realisticVisionV60B1_v51VAE
            |-- model_index.json
            |-- vae
            |-- ...
        |-- ...
```

The ckpt folder contains pretrinaed Stable Diffusion checkpoint (e.g., from [Civitai](https://civitai.com/)). We provide the processed realisticVisionV60B1_v51VAE ckpt [here](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n?usp=sharing). You can use `scripts/convert_original_stable_diffusion_to_diffusers.py` to process the checkpoint downloaded from Civitai.


Noted: *We only provide a part of the BrushData due to the space limit. Please write an email to juxuan.27@gmail.com if you need the full dataset.*



## 🏃🏼 Running Scripts


### Training 🤯

You can train with segmentation mask using the script:

```
accelerate launch examples/brushnet/train_brushnet.py \
--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_segmentationmask \
--train_data_dir data/BrushData \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300
```

To use custom dataset, you can process your own data to the format of BrushData and revise `--train_data_dir`.

You can train with random mask using the script (by adding `--random_mask`):

```
accelerate launch examples/brushnet/train_brushnet.py \
--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_randommask \
--train_data_dir data/BrushData \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300 \
--random_mask
```



### Inference 📜

You can inference with the script:

```
python examples/brushnet/test_brushnet.py
```

Remember to replace the `brushnet_path` with your local checkpoint path.


## 🤝🏼 Cite Us

```
@misc{ju2024brushnet,
  title={BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion}, 
  author={Xuan Ju and Xian Liu and Xintao Wang and Yuxuan Bian and Ying Shan and Qiang Xu},
  year={2024},
  eprint={2403.06976},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```


## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified based on [diffusers](https://github.com/huggingface/diffusers), thanks to all the contributors!

