from typing import Union
from typing import Optional
import torch
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import clip
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline

class MultiCLIP(torch.nn.Module):
    def __init__(self, device="cuda",dtype=torch.float32):
        super().__init__()
        model_32, _ = clip.load("../Tune-A-Video/ViT-B-32.pt", device=device)
        model_16, _ = clip.load("../Tune-A-Video/ViT-B-16.pt", device=device)
        model_101, _ = clip.load("../Tune-A-Video/RN101.pt", device=device)
        self.dtype=dtype
        self.model_32 = model_32
        self.model_16 = model_16
        self.model_101 = model_101
        self.preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    def encode_image(self, image, dtype=torch.float16):
        with torch.no_grad():
            image = self.preprocess(image)
            vectors = [self.model_16.encode_image(image.to(dtype)), self.model_32.encode_image(image.to(dtype)), self.model_101.encode_image(image.to(dtype))]
            #这里的vectors是一个list，里面有三个tensor，每个tensor的shape是(batch_size, 512)
            return torch.cat(vectors, dim=-1).to(self.dtype)
            #这里返回的是一个tensor，shape是(batch_size, 512*3)

    def encode_text(self, text, device,dtype=torch.float16,frame_size=8):
        with torch.no_grad():
            text = clip.tokenize(text).to(device)
            vectors = [self.model_16.encode_text(text), self.model_32.encode_text(text), self.model_101.encode_text(text)]
            vector_merge=torch.cat(vectors, dim=-1).to(dtype)
            vector_frame=vector_merge.repeat(frame_size,1)
            return vector_frame.to(self.dtype)
class CLIPLocalShift(torch.nn.Module):
    def __init__(self,uselocal=False):
        super().__init__()
        self.uselocal=uselocal
        if uselocal:
            self.img_layer_norm=torch.nn.LayerNorm([512*3])
            self.text_layer_norm=torch.nn.LayerNorm([512*3])
            self.cross_attn_layers=torch.nn.ModuleList([torch.nn.MultiheadAttention(512*3, 4) for _ in range(4)])
            self.linear=torch.nn.Linear(512*3, 4*64*64)
            self.out_norm=torch.nn.LayerNorm([4*64*64])
    def forward(self, image_emb, text_emb):
        img_norm=self.img_layer_norm(image_emb)
        text_norm=self.text_layer_norm(text_emb)
        x=img_norm
        for layer in self.cross_attn_layers:
            x,_=layer(x, text_norm, text_norm)
        x=self.linear(x)
        x=self.out_norm(x)
        return x
class DDPMTrainer(SpatioTemporalStableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        clip_model:Optional[MultiCLIP],
        local_shift:Optional[CLIPLocalShift],
        **kwargs
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
        )
        for name, module in kwargs.items():
            setattr(self, name, module)
        if clip_model is not None:
            self.clip_model=clip_model
        if local_shift is not None:
            self.local_shift=local_shift
    def step(self, batch: dict = dict(),prompt=None):
        if 'class_images' in batch:
            self.step2d(batch["class_images"], batch["class_prompt_ids"])
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        if self.prior_preservation is not None:
            print('Use prior_preservation loss')
            self.unet2d.eval()

        # Convert images to latent space
        images = batch["images"].to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        # print(images.shape)
        if prompt is not None:
            self.clip_model.eval()
            self.local_shift.train()
            image_embedding=self.clip_model.encode_image(images)
            text_embedding=self.clip_model.encode_text(prompt, device=self.device)
            local_shift=self.local_shift(image_embedding, text_embedding)
            local_shift=rearrange(local_shift, "( b f ) ( c h w ) -> b c f h w", b=1,c=4,f=8,h=64,w=64)
        # print(batch['prompt_ids'])
        # print(batch['prompt_ids'].shape)
        latents = self.vae.encode(images).latent_dist.sample() # shape=torch.Size([8, 3, 512, 512]), min=-1.00, max=0.98, var=0.21, -0.96875
        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b) #(1, 4, 8, 512, 512) 
        latents = latents * 0.18215
        
        # Sample noise that we'll add to the latents
        if prompt is not None:
            # print("Use prompt, use local shift")
            noise = torch.randn_like(latents)+local_shift*0.018125
        else:
            # print("No prompt, use random noise")
            noise=torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()
        # print(timesteps.shape)
        # print(timesteps)
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        # print(noisy_latents.shape)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["prompt_ids"])[0]
        # print(encoder_hidden_states.shape)

        # Predict the noise residual
        # if prompt is not None:
        #     model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states,context_awareness_shift=local_shift).sample
        # else:
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.prior_preservation is not None:
            model_pred_2d = self.unet2d(noisy_latents[:, :, 0], timesteps, encoder_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * self.prior_preservation
            )

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            if prompt is not None:
                self.accelerator.clip_grad_norm_(list(self.unet.parameters())+list(self.local_shift.parameters()), self.max_grad_norm)
            else:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    
    def step2d(self, class_images, prompt_ids
             ):
        
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.train()
        if self.prior_preservation is not None:
            self.unet2d.eval()


        # Convert images to latent space
        images = class_images.to(dtype=self.weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        latents = self.vae.encode(images).latent_dist.sample() # shape=torch.Size([8, 3, 512, 512]), min=-1.00, max=0.98, var=0.21, -0.96875
        
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(prompt_ids)[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.prior_preservation is not None:
            model_pred_2d = self.unet2d(noisy_latents[:, :, 0], timesteps, encoder_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * self.prior_preservation
            )

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss