## Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing

This repository contains the official implementation of text-to-image part in [ContextDiff](https://openreview.net/forum?id=nFMS6wF2xq). **Here, we only provide a sample code on COCO dataset for simplicity, and you may change to any datasets to apply our method.**


## Preparations

**Environment Setup**

```bash
git clone xxxx
conda create -n ContextDiff python==3.8
pip install -r requirements.txt
cd ContextDiff_image
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers
```

**Download Model Weights**

Here we choose [Stable Diffusion](https://arxiv.org/abs/2112.10752) as our diffusion backbone, you can download the model weights using our [download.py](ckpt/download.py) in folder 'ckpt/'. 

```bash
cd ckpt
python download.py 
wget "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"
wget "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
wget "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
cd ..
```


**Download Datasets**

```bash
cd datasets
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
cd ..
python process_img.py --src=./dataset/train2017 --size=512 --dest=./dataset/train2017
```

**Train Context-Aware Adapter**

```bash
CUDA_VISIBLE_DEVICES=0 python train_adapter.py --train_data_dir './dataset/train2017' --mixed_precision 'fp16' --output_dir 'output/' --train_batch_size 64 --num_train_epochs 20 --checkpointing_steps 10000 "--t5_model" 'path to text encoders' 
```

You can check the code for details, and choose hyper-parameters based on your device.

**Finetune Diffusion Model with Context-Aware Adapter**

```bash
CUDA_VISIBLE_DEVICES=0 finetune_diffusion.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --train_data_dir=./train2017 --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=32 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=50000 --checkpointing_steps=10000 --learning_rate=2e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 
--output_dir="./output"
```

For the '--mean_path' and '--std_path' in the code, it is generated from the dataset embeddings. You can use cluster method like GMM to obtain std and mean from your datasets.

## Citation
```
@inproceedings{
yang2024crossmodal,
title={Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing},
author={Ling Yang and Zhilong Zhang and Zhaochen Yu and Jingwei Liu and Minkai Xu and Stefano Ermon and Bin CUI},
booktitle={International Conference on Learning Representations},
year={2024}
}
```
