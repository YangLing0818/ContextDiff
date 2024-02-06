## Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing

This repository contains the official implementation of  [ContextDiff](https://openreview.net/forum?id=nFMS6wF2xq)

>[**Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing**](https://openreview.net/forum?id=nFMS6wF2xq)    
>[Ling Yang](https://yangling0818.github.io/), 
>[Zhilong Zhang](),
>[Zhaochen Yu](https://github.com/BitCodingWalkin), 
>[Jingwei Liu](),
>[Minkai Xu](https://minkaixu.com/),
>[Stefano Ermon](https://cs.stanford.edu/~ermon/), 
>[Bin Cui](https://cuibinpku.github.io/) 
<br>**Peking University, Stanford University**<br>

<details>
    <summary>Click for full abstract</summary>
    Conditional diffusion models have exhibited superior performance in high-fidelity
text-guided visual generation and editing. Nevertheless, prevailing text-guided visual diffusion models primarily focus on incorporating text-visual relationships
exclusively into the reverse process, often disregarding their relevance in the forward process. This inconsistency between forward and reverse processes may
limit the precise conveyance of textual semantics in visual synthesis results. To
address this issue, we propose a novel and general contextualized diffusion model
(CONTEXTDIFF) by incorporating the cross-modal context encompassing interactions and alignments between text condition and visual sample into forward and
reverse processes. We propagate this context to all timesteps in the two processes
to adapt their trajectories, thereby facilitating cross-modal conditional modeling.
We generalize our contextualized diffusion to both DDPMs and DDIMs with theoretical derivations, and demonstrate the effectiveness of our model in evaluations
with two challenging tasks: text-to-image generation, and text-to-video editing.
In each task, our CONTEXTDIFF achieves new state-of-the-art performance, significantly enhancing the semantic alignment between text condition and generated
samples, as evidenced by quantitative and qualitative evaluations.
</details>

## Introduction

<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="figs/Illustration.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Overview of our ContextDiff
</td>
  </tr>
</table>

We propose a novel and general cross-modal contextualized diffusion model (**ContextDiff**) that harnesses cross-modal context to facilitate the learning capacity of cross-modal diffusion models, including **text-to-image generation, and text-guided video editing**.

## 🚩 New Updates 

**[2024.1]** Our main code along with demo videos is released, including **text-to-image generation, and text-guided video editing**.


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
