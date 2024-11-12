import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import json
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import decord
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import torch.nn as nn
import diffusers
import transformers
import torchvision.transforms as transforms
# from utils.frechet_video_distance import frechet_video_distance as fvd
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from torcheval.metrics import FrechetInceptionDistance


from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

from common_metrics_on_video_quality.calculate_psnr import calculate_psnr
from models.fourier_embedding import FourierEmbedder
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

# from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing

from models.layerdiffuse_VAE import LatentSignalEncoder, SignalEncoder, SignalEncoder2, ImageReduction, \
    MultiSignalEncoder, TransformNet, FrameToSignalNet, SignalTransformer, CompactSignalEncoder2, \
    CompactSignalTransformer, CompactImageReduction, CompactSignalEncoder3, FFTConv1DLinearModel, \
    CompactSignalTransformer2, CompactSignalEncoder3_2, CompactImageReduction2
# from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from utils.dataset import get_train_dataset, extend_datasets, normalize_input
from einops import rearrange, repeat
import imageio
import wandb
# from models.unet_3d_condition import UNet3DConditionModel
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from utils.pips.demo import run_model
from utils.pips.nets.pips import Pips
from models.pipeline_signal_vanila import MaskStableVideoDiffusionPipeline

decord.bridge.set_bridge('torch')

already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir


def load_primary_models(pretrained_model_path, fps, frame_step, n_input_frames, width, height, eval=False):
    # 25 = 4(latent/noisy) + 1(signal) // + n_input_frames(5) // 1(initial signal)
    # prev in_channels: cond(4) + noise(4) (+ mask(1))
    # ++ init_images(1) + init_signals(1) + signal(1) + pos(1)
    in_channels = 8
    if eval:
        pipeline = MaskStableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16,
                                                                    variant='fp16')
    else:
        pipeline = MaskStableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)

    if in_channels > 0 and pipeline.unet.config.in_channels != in_channels:
        prev_channel = pipeline.unet.config.in_channels
        # first time init, modify unet conv in
        unet2 = pipeline.unet
        unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_path + "/unet",
                                                                in_channels=in_channels,
                                                                low_cpu_mem_usage=False, device_map=None,
                                                                ignore_mismatched_sizes=True)
        unet.conv_in.bias.data = copy.deepcopy(unet2.conv_in.bias)
        torch.nn.init.zeros_(unet.conv_in.weight)
        load_in_channel = unet2.conv_in.weight.data.shape[1]
        unet.conv_in.weight.data[:, in_channels - load_in_channel:] = copy.deepcopy(unet2.conv_in.weight.data)
        pipeline.unet = unet

        print(f"#########Unet channel is changed from {prev_channel} to {pipeline.unet.config.in_channels} ########")
        del unet2
    prev_channel = pipeline.unet.config.in_channels
    # first time init, modify unet conv in
    unet2 = pipeline.unet
    unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_path + "/unet",
                                                            in_channels=in_channels,
                                                            low_cpu_mem_usage=False, device_map=None,
                                                            ignore_mismatched_sizes=True)
    unet.conv_in.bias.data = copy.deepcopy(unet2.conv_in.bias)
    torch.nn.init.zeros_(unet.conv_in.weight)
    load_in_channel = unet2.conv_in.weight.data.shape[1]
    unet.conv_in.weight.data[:, in_channels - load_in_channel:] = copy.deepcopy(unet2.conv_in.weight.data)
    pipeline.unet = unet

    print(f"#########Unet channel is changed from {prev_channel} to {pipeline.unet.config.in_channels} ########")
    del unet2
    CHIRP_LEN = 512
    encoder_hidden_dim = 1024

    # signal_encoder = LatentSignalEncoder(output_dim=encoder_hidden_dim)
    # signal_encoder = SignalEncoder(input_size=CHIRP_LEN, frame_step=2, output_size=encoder_hidden_dim)
    image_encoder = CompactImageReduction2(input_dim=4, frame_step=frame_step, n_input_frames=n_input_frames, target_h=width // 8, target_w=height // 8)

    # for intiial signal
    # n_input_frames += 1
    fps += 1
    # signal_encoder = FrameToSignalNet(input_size=CHIRP_LEN, n_input_frames=n_input_frames, frame_step=frame_step, output_size=encoder_hidden_dim)
    signal_encoder = CompactSignalTransformer2(input_size=CHIRP_LEN, output_dim=1, frame_step=frame_step, n_input_frames=n_input_frames, target_h=16, target_w=64)


    # Just large dim for later interpolation
    input_latents_dim1 = 100
    input_latents_dim2 = 100

    # signal_encoder2 = LatentSignalEncoder(output_dim=input_latents_dim1 * input_latents_dim2)
    signal_encoder2 = CompactSignalEncoder3_2(signal_data_dim=CHIRP_LEN, output_dim=1, fps=fps, frame_step=frame_step, target_h=width // 8, target_w=height // 8)
    signal_encoder3 = CompactSignalTransformer2(input_size=CHIRP_LEN, output_dim=1, frame_step=frame_step, n_input_frames=n_input_frames, target_h=width // 8, target_w=height // 8)

    # Embed specific AP's location as bounding box
    # x_min, y_min, z_min, x_max, y_max, z_max
    # fourier_freqs*2*6: fourier_embedder's output shape (2 is sin&cos, 6 is xyzxyz)
    # 3 x 8 x 2 = 48
    # 4 x 8 x 2 = 64
    fourier_freqs = 8
    camera_fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs, output_dim=4*4*fourier_freqs*2, temperature=2, target_h=width // 16, target_w=height // 16)
    tx_fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs, output_dim=3*fourier_freqs*2, temperature=2, target_h=width // 16, target_w=height // 16)

    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.image_processor, \
           pipeline.image_encoder, pipeline.vae, pipeline.unet, signal_encoder, signal_encoder2, signal_encoder3,\
           camera_fourier_embedder, tx_fourier_embedder, image_encoder


def convert_svd(pretrained_model_path, out_path):
    pipeline = MaskStableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet_mask", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    unet.conv_in.bias.data = copy.deepcopy(pipeline.unet.conv_in.bias)
    torch.nn.init.zeros_(unet.conv_in.weight)
    unet.conv_in.weight.data[:, 1:] = copy.deepcopy(pipeline.unet.conv_in.weight)
    new_pipeline = MaskStableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path, unet=unet)
    new_pipeline.save_pretrained(out_path)


def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    if unet_enable:
        unet.enable_gradient_checkpointing()
    else:
        unet.disable_gradient_checkpointing()
    if text_enable:
        text_encoder.gradient_checkpointing_enable()
    else:
        text_encoder.gradient_checkpointing_disable()


def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False)


def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])


def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)

    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")


def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }


def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        for n, p in model.named_parameters():
            if p.requires_grad:
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params


def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW


def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype


def cast_to_gpu_and_type(model_list, device, weight_dtype):
    for model in model_list:
        if model is not None: model.to(device, dtype=weight_dtype)


def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break

                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params += 1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")


def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents


def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
            alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 5) \
           and validation_data.sample_preview


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        sig1,
        sig2,
        sig3,
        camera_fourier,
        tx_fourier,
        img1,
        output_dir,
        is_checkpoint=False,
        save_pretrained_model=True
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    unet_out = copy.deepcopy(unet)
    pipeline = MaskStableVideoDiffusionPipeline.from_pretrained(
        path, unet=unet_out).to(torch_dtype=torch.float32)

    sig1_out = copy.deepcopy(sig1)
    sig2_out = copy.deepcopy(sig2)
    sig3_out = copy.deepcopy(sig3)

    img1_out = copy.deepcopy(img1)

    camera_fourier_out = copy.deepcopy(camera_fourier)
    tx_fourier_out = copy.deepcopy(tx_fourier)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)
        signal_save_path = save_path + "/signal/"
        os.makedirs(signal_save_path, exist_ok=True)
        torch.save(sig1_out.state_dict(), signal_save_path + 'sig1.pth')
        torch.save(sig2_out.state_dict(), signal_save_path + 'sig2.pth')
        torch.save(sig3_out.state_dict(), signal_save_path + 'sig3.pth')
        torch.save(img1_out.state_dict(), signal_save_path + 'img1.pth')

        torch.save(camera_fourier_out.state_dict(), signal_save_path + 'camera_fourier.pth')
        torch.save(tx_fourier_out.state_dict(), signal_save_path + 'tx_fourier.pth')

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del sig1_out, sig2_out, sig3_out, img1_out,
    torch.cuda.empty_cache()
    gc.collect()


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt


def prompt_image(image, processor, encoder):
    if type(image) == str:
        image = Image.open(image)
    image = processor(images=image, return_tensors="pt")['pixel_values']

    image = image.to(encoder.device).to(encoder.dtype)
    inputs = encoder(image).pooler_output.to(encoder.dtype).unsqueeze(1)
    # inputs = encoder(image).last_hidden_state.to(encoder.dtype)
    return inputs


def finetune_unet(accelerator, pipeline, batch, use_offset_noise,
                  rescale_schedule, offset_noise_strength, unet, sig1, sig2, sig3, camera_fourier, tx_fourier, img1, n_input_frames, motion_mask,
                  P_mean=0.7, P_std=1.6):
    pipeline.vae.eval()
    pipeline.image_encoder.eval()

    device = unet.device
    dtype = pipeline.vae.dtype
    vae = pipeline.vae
    # Convert videos to latent space
    pixel_values = batch['pixel_values']
    bsz, num_frames = pixel_values.shape[:2]

    frames = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)
    latents = vae.encode(frames).latent_dist.mode() * vae.config.scaling_factor
    latents = rearrange(latents, '(b f) c h w-> b f c h w', b=bsz)  # 1 Channel

    image_pool = img1.to(latents.device).to(dtype)

    # enocde image latent
    image = pixel_values[:, 0].to(dtype)
    noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
    image = image + noise_aug_strength * torch.randn_like(image)
    image_latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor  # # n_input_frames Channel
    condition_latent = repeat(image_latent, 'b c h w->b f c h w', f=num_frames)

    image = pixel_values[:, 0:n_input_frames].to(dtype)
    image = rearrange(image, 'b f c h w-> (b f) c h w').to(dtype)
    # noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
    # image = image + noise_aug_strength * torch.randn_like(image)

    image_latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor  # # n_input_frames Channel
    image_latent = rearrange(image_latent, '(b f) c h w-> b f c h w', b=bsz).to(dtype)
    # torch.Size([2, 10, 4, 64, 64])
    image_latent = image_pool(image_latent) / vae.config.scaling_factor
    # images_latent = image_latent.repeat(1, num_frames, 1, 1, 1)

    pipeline.image_encoder.to(device, dtype=dtype)

    # print("image_embedding: ", image_embeddings.size()) image_embedding:  torch.Size([2, 1, 1024])
    rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    c_skip = 1 / (sigma ** 2 + 1)
    c_out = -sigma / (sigma ** 2 + 1) ** 0.5
    c_in = 1 / (sigma ** 2 + 1) ** 0.5
    c_noise = (sigma.log() / 4).reshape([bsz])
    loss_weight = (sigma ** 2 + 1) / sigma ** 2

    noisy_latents = latents + torch.randn_like(latents) * sigma
    input_latents = torch.cat([c_in * noisy_latents, condition_latent / vae.config.scaling_factor], dim=2)

    # Signal embedding
    assert "frame_step" in batch.keys(), batch.keys()
    frame_step = batch["frame_step"][0].item()
    # print("frame_step", frame_step)

    image_latent = image_latent.reshape(bsz, 1, -1)

    encoder_hidden_states = image_latent
    uncond_hidden_states = torch.zeros_like(encoder_hidden_states)

    if random.random() < 0.15:
        encoder_hidden_states = uncond_hidden_states

    encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process) #[bsz, f, c, h , w]

    # input_latents:  torch.Size([2, 25, 8, 64, 64])
    # signal_embeddings2: torch.Size([2, 25, 8 -> 1, 64, 64])
    # print("signal_embeddings2", signal_embeddings2.size())
    # print("final latents ", input_latents.size()) # final latents  torch.Size([2, 25, 9, 64, 64])


    # tx_latent = tx_latent.repeat(1, num_frames, 1, 1, 1)  # condition_latent torch.Size([1, 50, 20, 8, 8])
    # pos_latent = torch.cat((camera_latent, tx_latent), dim=3)
    motion_bucket_id = 127
    fps = 6
    added_time_ids = pipeline._get_add_time_ids(fps, motion_bucket_id,
                                                noise_aug_strength, dtype, bsz, 1, False)
    added_time_ids = added_time_ids.to(device)

    loss = 0
    # print(signal_initial_latent.size(), signal_latent.size(), images_latent.size(), input_latents.size())
    # torch.Size([2, 25, 1, 64, 64]) torch.Size([2, 25, 1, 64, 64]) torch.Size([2, 25, 5, 64, 64]) torch.Size([2, 25, 8, 64, 64])
    # latent_model_input = torch.cat([input_latents], dim=2)
    latent_model_input = input_latents
    accelerator.wait_for_everyone()
    # print(input_latents.size(), c_noise.size(), encoder_hidden_states.size(), added_time_ids.size())
    # torch.Size([2, 25, 9, 1, 1]) torch.Size([2]) torch.Size([50, 1, 1024]) torch.Size([2, 3])
    model_pred = unet(latent_model_input, c_noise, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids).sample
    predict_x0 = c_out * model_pred + c_skip * noisy_latents
    loss += ((predict_x0 - latents) ** 2 * loss_weight).mean()
    return loss


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        extra_train_data: list = [],
        dataset_types: Tuple[str] = 'json',
        shuffle: bool = True,
        validation_steps: int = 100,
        trainable_modules: Tuple[str] = None,  # Eg: ("attn1", "attn2")
        extra_unet_params=None,
        extra_text_encoder_params=None,
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 5e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        text_encoder_gradient_checkpointing: bool = False,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        resume_step: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        enable_torch_2_attn: bool = False,
        seed: Optional[int] = None,
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        offset_noise_strength: float = 0.1,
        extend_dataset: bool = False,
        cache_latents: bool = False,
        cached_latent_dir=None,
        save_pretrained_model: bool = True,
        logger_type: str = 'tensorboard',
        motion_mask=False,
        **kwargs
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir,
        # project_config=config,
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models. The text encoder is actually image encoder for SVD
    pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet, sig1, sig2, \
    sig3, camera_fourier, tx_fourier, img1 = load_primary_models(
        pretrained_model_path, train_data.n_sample_frames, train_data.frame_step, train_data.n_input_frames, train_data.width, train_data.height)
    # Freeze any necessary models
    freeze_models([vae, unet])

    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    trainable_modules_available = trainable_modules is not None

    # Unfreeze UNET Layers
    if trainable_modules_available:
        unet.train()
        handle_trainable_modules(
            unet,
            trainable_modules,
            is_enabled=True,
        )

    optim_params = [
        param_optim(unet, trainable_modules_available, extra_params=extra_unet_params)
    ]

    params = create_optimizer_params(optim_params, learning_rate)

    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Get the training dataset based on types (json, single_video, image)
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    # If you have extra train data, you can add a list of however many you would like.
    # Eg: extra_train_data: [{: {dataset_types, train_data: {etc...}}}]
    try:
        if extra_train_data is not None and len(extra_train_data) > 0:
            for dataset in extra_train_data:
                d_t, t_d = dataset['dataset_types'], dataset['train_data']
                train_datasets += get_train_dataset(d_t, t_d, tokenizer)

    except Exception as e:
        print(f"Could not process extra train datasets due to an error : {e}")

    # Extend datasets that are less than the greatest one. This allows for more balanced training.
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    # Process one dataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]

    # Process many datasets
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

        # DataLoaders creation:

    # Define the split sizes
    n = 50
    interval = len(train_dataset) // n
    test_dataset = [train_dataset[i] for i in range(0, len(train_dataset), interval)][:n]


    # Split the dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle
    )
    evaluator = None
    if accelerator.is_main_process:
        from utils.cdfvd import fvd
        evaluator = fvd.cdfvd('videomae', ckpt_path='vit_g_hybrid_pt_1200e_ssv2_ft', n_fake='full')
        if not os.path.exists("fvd_real.stat"):
            real_videos = []
            for step, batch in enumerate(tqdm(train_dataloader)):
                if step >= 500:
                    break
                pixel_values_real = batch['pixel_values_real']
                """
                pixel_values = batch['pixel_values']
                image_path = batch['pixel_values_path']
                bsz, num_frames = pixel_values.shape[:2]
                vr = decord.VideoReader(image_path[0])
                frame_step = 3
                frame_range = list(range(0, len(vr), frame_step))
                frames = vr.get_batch(frame_range[0:validation_data.num_frames])
                frames = rearrange(frames, "f h w c -> f c h w")
                # frames = frames.cpu().numpy()  # Convert to a NumPy array if it's a tensor
                real_videos.append(frames)
                transform = T.Compose([
                    # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
                    T.Resize(min(448, 448), antialias=False),
                    T.CenterCrop([448, 448])
                ])
                frames = transform(frames)
                print(pixel_values_real.size(), frames.size())
                assert torch.equal(pixel_values_real[0], frames), (pixel_values_real[0], frames)
                print("good")
                """
                real_videos.append(pixel_values_real)
            real_videos = torch.stack(real_videos)
            real_videos = rearrange(real_videos, "b1 b2 f c h w -> (b1 b2) f c h w")
            evaluator.compute_real_stats(evaluator.load_videos(".pt", data_type="video_torch", video_data=real_videos))
            evaluator.save_real_stats("fvd_real.stat")
            # # Shape: (1, T, C, H, W)
            del real_videos
        else:
            evaluator.load_real_stats("fvd_real.stat")

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet,
        text_encoder,
        gradient_checkpointing,
        text_encoder_gradient_checkpointing
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae, sig1, sig2, sig3, camera_fourier, tx_fourier, img1]
    cast_to_gpu_and_type(models_to_cast, accelerator.device, weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("vanila_empty0.05_metrics_videomae")
        wandb.login(key="a94ace7392048e560ce6962a468101c6f0158b55")
        wandb.require("core")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    num_train_epochs = math.ceil(
        max_train_steps * gradient_accumulation_steps / len(train_dataloader) / accelerator.num_processes)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # *Potentially* Fixes gradient checkpointing training.
    # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    if kwargs.get('eval_train', False):
        unet.eval()
        text_encoder.eval()
    """
    if accelerator.is_main_process:
        print("Save Test")
        save_pipe(
            pretrained_model_path,
            global_step,
            accelerator,
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(text_encoder),
            vae,
            accelerator.unwrap_model(sig1),
            accelerator.unwrap_model(sig2),
            output_dir,
            is_checkpoint=True,
            save_pretrained_model=save_pretrained_model
        )
    """
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(sig1), accelerator.accumulate(
                    sig2), accelerator.accumulate(sig3), accelerator.accumulate(camera_fourier), \
                    accelerator.accumulate(tx_fourier), accelerator.accumulate(img1):
                with accelerator.autocast():
                    loss = finetune_unet(accelerator, pipeline, batch, use_offset_noise,
                                         rescale_schedule, offset_noise_strength, unet, sig1, sig2, sig3, camera_fourier,
                                         tx_fourier, img1,
                                         train_data.n_input_frames, motion_mask)
                device = loss.device
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                params_to_clip = unet.parameters()
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                global_step += 1
                train_loss = 0.0

                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        accelerator,
                        accelerator.unwrap_model(unet),
                        accelerator.unwrap_model(text_encoder),
                        vae,
                        accelerator.unwrap_model(sig1),
                        accelerator.unwrap_model(sig2),
                        accelerator.unwrap_model(sig3),
                        accelerator.unwrap_model(camera_fourier),
                        accelerator.unwrap_model(tx_fourier),
                        accelerator.unwrap_model(img1),
                        output_dir,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data) and accelerator.is_main_process:
                    if global_step == 1: print("Performing validation prompt.")
                    with accelerator.autocast():
                        curr_dataset_name = batch['dataset'][0]
                        save_filename = f"{global_step}_dataset-{curr_dataset_name}"
                        out_file = f"{output_dir}/samples/"
                        eval(pipeline, vae_processor, sig1, sig2, sig3, camera_fourier, tx_fourier, img1, validation_data, out_file, global_step)
                        logger.info(f"Saved a new sample to {out_file}")
                        if global_step > 8000:
                            fvd = eval_fid_fvd_videomae(evaluator, test_dataloader, pipeline, vae_processor,
                                                        sig1, sig2,
                                                        sig3, camera_fourier, tx_fourier, img1, None,
                                                        validation_data, out_file, global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
            pretrained_model_path,
            global_step,
            accelerator,
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(text_encoder),
            vae,
            accelerator.unwrap_model(sig1),
            accelerator.unwrap_model(sig2),
            accelerator.unwrap_model(sig3),
            accelerator.unwrap_model(camera_fourier),
            accelerator.unwrap_model(tx_fourier),

            accelerator.unwrap_model(img1),

            output_dir,
            is_checkpoint=False,
            save_pretrained_model=save_pretrained_model
        )
    accelerator.end_training()


def eval(pipeline, vae_processor, sig1, sig2, sig3, camera_fourier, tx_fourier, img1, validation_data, out_file, index, forward_t=25, preview=True):
    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype

    diffusion_scheduler = pipeline.scheduler
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)

    # prompt = validation_data.prompt
    # scale = math.sqrt(width * height / (validation_data.height * validation_data.width))
    # block_size = 64
    # validation_data.height = round(height / scale / block_size) * block_size
    # validation_data.width = round(width / scale / block_size) * block_size

    # out_mask_path = os.path.splitext(out_file)[0] + "_mask.jpg"
    # Image.fromarray(np_mask).save(out_mask_path)
    motion_mask = True
    # prepare inital latents
    initial_latents = None
    transform = T.Compose([
        # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
        T.Resize(min(validation_data.height, validation_data.width), antialias=False),
        T.CenterCrop([validation_data.height, validation_data.width])
    ])

    for image in sorted(validation_data.prompt_image):
        # print(out_file)
        # print(image)
        signal = image.replace(".mp4", ".pt")
        initial_signal = signal.replace(".pt", "_init.pt")

        camera_pose = image.replace(".mp4", ".npy")
        tx_loc = image.replace(".mp4", ".txt")

        image_replaced = image.replace("frame", str(index) + "_frame").replace('.mp4', '.gif')
        target_file = out_file + image_replaced
        # print(out_file)
        # print(image_replaced)
        directory = os.path.dirname(target_file)
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # pimg = Image.open(image)
        vr = decord.VideoReader(image)
        frame_step = 3
        frame_range = list(range(0, len(vr), frame_step))
        frames = vr.get_batch(frame_range[0:validation_data.num_frames])

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()  # Convert to a NumPy array if it's a tensor

        # Convert each frame to a PIL.Image
        pil_images = []
        for i in range(frames.shape[0]):  # Iterate over the batch
            frame = frames[i]  # Get the i-th frame, shape (height, width, 3)
            pil_image = Image.fromarray(frame)  # Convert to PIL.Image
            pil_images.append(pil_image)
        # video = rearrange(frames, "f h w c -> f c h w")
        # video = transform(video)
        # video = normalize_input(video)

        signal = torch.real(torch.load(signal, map_location="cuda:0", weights_only=True)).to(dtype).to(device)
        initial_signal = torch.real(torch.load(initial_signal, map_location="cuda:0", weights_only=True)).to(dtype).to(device)
        initial_channels = initial_signal.unsqueeze(0)  # Now shape is (1, 512)

        # result_signal = torch.cat((initial_channels, signal), dim=0)  # Result shape will be (53, 512)
        result_signal = signal - initial_channels

        # result_signal = result_signal * 1e4
        # result_signal = log_scale_tensor(torch.abs(result_signal))
        if torch.isnan(result_signal).any():
            print(result_signal)
            result_signal = torch.nan_to_num(result_signal, nan=0.0)

        camera_data = np.load(camera_pose)
        tx_data = np.loadtxt(tx_loc)

        camera_data = torch.from_numpy(camera_data).to(dtype).to(device)
        tx_data = torch.from_numpy(tx_data).to(dtype).to(device)

        with torch.no_grad():
            if motion_mask:
                # h, w = validation_data.height // pipeline.vae_scale_factor, validation_data.width // pipeline.vae_scale_factor
                # initial_latents = torch.randn([1, validation_data.num_frames, 4, h, w], dtype=dtype, device=device)
                # mask = T.ToTensor()(np_mask).to(dtype).to(device)
                # mask = T.Resize([h, w], antialias=False)(mask)
                video_frames = MaskStableVideoDiffusionPipeline.__call__(
                    pipeline,
                    video=pil_images,
                    width=validation_data.width,
                    height=validation_data.height,
                    num_frames=validation_data.num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    decode_chunk_size=validation_data.decode_chunk_size,
                    fps=validation_data.fps,
                    motion_bucket_id=validation_data.motion_bucket_id,
                    n_input_frames=validation_data.n_input_frames,
                    signal_latent=None,
                    signal=result_signal,
                    camera_pose=camera_data,
                    tx_pos=tx_data,
                    sig1=sig1,
                    sig2=sig2,
                    sig3=sig3,
                    camera_fourier=camera_fourier,
                    tx_fourier=tx_fourier,
                    img1=img1,
                ).frames[0]
            else:
                video_frames = pipeline(
                    video=pil_images,
                    width=validation_data.width,
                    height=validation_data.height,
                    num_frames=validation_data.num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    fps=validation_data.fps,
                    decode_chunk_size=validation_data.decode_chunk_size,
                    motion_bucket_id=validation_data.motion_bucket_id,
                ).frames[0]

        if preview:
            fps = validation_data.get('fps', 8)
            imageio.mimwrite(target_file, video_frames, duration=int(1000 / fps), loop=0)
            imageio.mimwrite(target_file.replace('.gif', '.mp4'), video_frames, fps=fps)
            # resized_frames = [np.array(cv2.resize(frame, (125, 125))) for frame in np.array(video_frames)]
            # resized_frames = np.array(resized_frames)
            wandb.log({image: wandb.Video(target_file.replace('.gif', '.mp4'),
                                          caption=target_file.replace('.gif', '.mp4'), format="mp4")})

    return 0


def eval_fid_fvd(test_dataloader, pipeline, vae_processor, sig1, sig2, sig3, camera_fourier, tx_fourier, img1, validation_data, out_file, index, forward_t=25, preview=True):
    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype
    fid_results = []
    fvd_results = []

    diffusion_scheduler = pipeline.scheduler
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)
    target_file = out_file + "test.mp4"
    # print(out_file)
    # print(image_replaced)
    directory = os.path.dirname(target_file)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # prompt = validation_data.prompt
    # scale = math.sqrt(width * height / (validation_data.height * validation_data.width))
    # block_size = 64
    # validation_data.height = round(height / scale / block_size) * block_size
    # validation_data.width = round(width / scale / block_size) * block_size

    # out_mask_path = os.path.splitext(out_file)[0] + "_mask.jpg"
    # Image.fromarray(np_mask).save(out_mask_path)
    videos1 = []
    videos2 = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch['pixel_values'].to(dtype)
        image_path = batch['pixel_values_path']
        bsz, num_frames = pixel_values.shape[:2]
        vr = decord.VideoReader(image_path[0])
        frame_step = 3
        frame_range = list(range(0, len(vr), frame_step))
        frames = vr.get_batch(frame_range[0:validation_data.num_frames])

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()  # Convert to a NumPy array if it's a tensor

        # Convert each frame to a PIL.Image
        pil_images = []
        for i in range(frames.shape[0]):  # Iterate over the batch
            frame = frames[i]  # Get the i-th frame, shape (height, width, 3)
            pil_image = Image.fromarray(frame)  # Convert to PIL.Image
            pil_images.append(pil_image)

        result_signal = torch.real(batch['signal_values']).float().half().squeeze(0)
        result_signal = result_signal * 1e3
        if torch.isnan(result_signal).any():
            print(result_signal)
            result_signal = torch.nan_to_num(result_signal, nan=0.0)

        # camera_data = np.load(camera_pose)
        # tx_data = np.loadtxt(tx_loc)

        camera_data = batch['camera_pose'].float().half().to(device).squeeze(0)
        tx_data = batch['tx_pos'].float().half().to(dtype).to(device).squeeze(0)

        with torch.no_grad():
            # h, w = validation_data.height // pipeline.vae_scale_factor, validation_data.width // pipeline.vae_scale_factor
            # initial_latents = torch.randn([1, validation_data.num_frames, 4, h, w], dtype=dtype, device=device)
            # mask = T.ToTensor()(np_mask).to(dtype).to(device)
            # mask = T.Resize([h, w], antialias=False)(mask)
            video_frames = MaskStableVideoDiffusionPipeline.__call__(
                pipeline,
                video=pil_images,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=validation_data.num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                decode_chunk_size=validation_data.decode_chunk_size,
                fps=validation_data.fps,
                motion_bucket_id=validation_data.motion_bucket_id,
                n_input_frames=validation_data.n_input_frames,
                signal_latent=None,
                signal=result_signal,
                camera_pose=camera_data,
                tx_pos=tx_data,
                sig1=sig1,
                sig2=sig2,
                sig3=sig3,
                camera_fourier=camera_fourier,
                tx_fourier=tx_fourier,
                img1=img1,
            ).frames[0]

        transform = transforms.ToTensor()

        # Apply the transformation to each image in the list and collect tensors
        # tensor_list = [transform(img) for img in pil_images]
        # print("0", video_frames)

        # Stack the tensors into a single tensor (batch of images)
        transform = transforms.ToTensor()
        # Convert each image in the list to a tensor
        video_frames = [transform(image) for image in video_frames]
        # Stack the list of tensors into a single tensor
        video_frames = torch.stack(video_frames)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        # print("2", pil_images[0])
        transform = T.Compose([
            # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
            T.Resize(min(validation_data.height, validation_data.width), antialias=False),
            T.CenterCrop([validation_data.height, validation_data.width]),
            T.ToTensor()
        ])
        pil_images = [transform(image) for image in pil_images]

        # Stack the list of tensors into a single tensor
        pil_images = torch.stack(pil_images)

        pil_images = pil_images.squeeze(1)
        if pil_images.size(0) < video_frames.size(0):
            last_element = pil_images[-1].unsqueeze(0)  # Shape [1, 3, 64, 64]

            # Repeat the last element to fill up the desired shape
            repeated_elements = last_element.repeat(video_frames.size(0)-pil_images.size(0), 1, 1, 1)  # Repeat 3 times

            # Concatenate the original tensor with the repeated elements
            pil_images = torch.cat((pil_images, repeated_elements), dim=0)  # Shape [25, 3, 64, 64]

        # fvd_result = calculate_fvd(pil_images.unsqueeze(0), video_frames.unsqueeze(0), device, method='styleganv')
        # fvd_result = calculate_fvd(videos1, videos2, device, method='styleganv')
        # fvd_result2 = calculate_fvd(pil_images, video_frames, device, method='styleganv')
        videos1.append(pil_images)
        videos2.append(video_frames)
        # print(pil_images)
        # print(video_frames)
        # print(fvd_result, fvd_result2)
        # fvd_results.append(fvd_result)
        # video_tensor = pil_images.squeeze(0)

        # Convert to NumPy array and change the order of dimensions [frames, channels, height, width] -> [frames, height, width, channels]
        # video_numpy = video_tensor.permute(0, 2, 3, 1).numpy()

        # Normalize pixel values to the range [0, 255]
        # video_numpy = (255 * (video_numpy - video_numpy.min()) / (video_numpy.max() - video_numpy.min())).astype(np.uint8)

        # imageio.mimwrite(target_file, video_numpy, fps=fps)

    # NUMBER_OF_VIDEOS = 8
    # VIDEO_LENGTH = 30
    # CHANNEL = 3
    # SIZE = 64
    # tmp_1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    # tmp_2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos1 = torch.stack(videos1)  # torch.Size([11, 25, 3, 64, 64]) torch.Size([11, 25, 3, 64, 64])
    videos2 = torch.stack(videos2)
    # print(videos1.min(), videos2.min(), videos1.max(), videos2.max())
    # print(videos1.size(), videos2.size())
    # fvd_result = calculate_fvd(videos1, videos2, device, method='videogpt')
    fvd_result = calculate_fvd(videos1, videos2, device, method='styleganv')

    psnr_result = calculate_psnr(videos1, videos2)

    # print(fvd_result)
    # print(fvd_result2)
    # print(fvd_result)
    fvd_avg = []
    for val_key in fvd_result['value'].keys():
        fvd_avg.append(fvd_result['value'][val_key])
    fvd_results = fvd_result['value'][25]
    fvd_avg = np.sum(fvd_avg) / len(fvd_avg)

    psnr_avg = []
    for val_key in psnr_result['value'].keys():
        psnr_avg.append(psnr_result['value'][val_key])
    psnr_avg = np.sum(psnr_avg) / len(psnr_avg)

    fid = FrechetInceptionDistance(device='cuda')

    # Simulate loading batches of real and generated images
    for video_idx in range(len(videos1)):
        # Generate dummy data for demonstration (replace these with your real data loading)
        real_images = videos1[video_idx].to('cuda')  # Replace with your real images
        generated_images = videos2[video_idx].to('cuda')  # Replace with your generated images

        # Update the FID metric with real and generated images
        fid.update(real_images, is_real=True)
        fid.update(generated_images, is_real=False)

    # Compute the FID score after processing all batches
    fid_score = fid.compute()

    # fid_results = np.sum(fid_avg) / len(fid_avg)
    wandb.log({"fid": fid_score})
    wandb.log({"fvd": fvd_results})
    wandb.log({"fvd (avg)": fvd_avg})
    wandb.log({"psnr": psnr_avg})

    return fid_score, fvd_results, fvd_avg, psnr_avg


def eval_fid_fvd_videomae(evaluator, test_dataloader, pipeline, vae_processor, sig1, sig2, sig3, camera_fourier, tx_fourier, img1, final_encoder, validation_data, out_file, index, forward_t=25, preview=True):
    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype
    fid_results = []
    fvd_results = []

    diffusion_scheduler = pipeline.scheduler
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)
    target_file = out_file + "test.mp4"
    # print(out_file)
    # print(image_replaced)
    directory = os.path.dirname(target_file)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # prompt = validation_data.prompt
    # scale = math.sqrt(width * height / (validation_data.height * validation_data.width))
    # block_size = 64
    # validation_data.height = round(height / scale / block_size) * block_size
    # validation_data.width = round(width / scale / block_size) * block_size

    # out_mask_path = os.path.splitext(out_file)[0] + "_mask.jpg"
    # Image.fromarray(np_mask).save(out_mask_path)
    videos1 = []
    videos2 = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch['pixel_values'].to(dtype)
        image_path = batch['pixel_values_path']
        bsz, num_frames = pixel_values.shape[:2]
        vr = decord.VideoReader(image_path[0])
        frame_step = 3
        frame_range = list(range(0, len(vr), frame_step))
        frames = vr.get_batch(frame_range[0:validation_data.num_frames])

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()  # Convert to a NumPy array if it's a tensor

        # Convert each frame to a PIL.Image
        pil_images = []
        for i in range(frames.shape[0]):  # Iterate over the batch
            frame = frames[i]  # Get the i-th frame, shape (height, width, 3)
            pil_image = Image.fromarray(frame)  # Convert to PIL.Image
            pil_images.append(pil_image)

        result_signal = torch.real(batch['signal_values']).float().half().squeeze(0)
        result_signal = result_signal * 1e3
        if torch.isnan(result_signal).any():
            print(result_signal)
            result_signal = torch.nan_to_num(result_signal, nan=0.0)

        # camera_data = np.load(camera_pose)
        # tx_data = np.loadtxt(tx_loc)

        camera_data = batch['camera_pose'].float().half().to(device).squeeze(0)
        tx_data = batch['tx_pos'].float().half().to(dtype).to(device).squeeze(0)

        with torch.no_grad():
            # h, w = validation_data.height // pipeline.vae_scale_factor, validation_data.width // pipeline.vae_scale_factor
            # initial_latents = torch.randn([1, validation_data.num_frames, 4, h, w], dtype=dtype, device=device)
            # mask = T.ToTensor()(np_mask).to(dtype).to(device)
            # mask = T.Resize([h, w], antialias=False)(mask)
            video_frames = MaskStableVideoDiffusionPipeline.__call__(
                pipeline,
                video=pil_images,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=validation_data.num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                decode_chunk_size=validation_data.decode_chunk_size,
                fps=validation_data.fps,
                motion_bucket_id=validation_data.motion_bucket_id,
                n_input_frames=validation_data.n_input_frames,
                signal_latent=None,
                signal=result_signal,
                camera_pose=camera_data,
                tx_pos=tx_data,
                sig1=sig1,
                sig2=sig2,
                sig3=sig3,
                camera_fourier=camera_fourier,
                tx_fourier=tx_fourier,
                img1=img1,
            ).frames[0]

        transform = transforms.ToTensor()

        # Apply the transformation to each image in the list and collect tensors
        # tensor_list = [transform(img) for img in pil_images]
        # print("0", video_frames)

        # Stack the tensors into a single tensor (batch of images)
        transform = transforms.ToTensor()
        # Convert each image in the list to a tensor
        video_frames = [transform(image) for image in video_frames]
        # Stack the list of tensors into a single tensor
        video_frames = torch.stack(video_frames)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        # print("2", pil_images[0])
        transform = T.Compose([
            # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
            T.Resize(min(validation_data.height, validation_data.width), antialias=False),
            T.CenterCrop([validation_data.height, validation_data.width]),
            T.ToTensor()
        ])
        pil_images = [transform(image) for image in pil_images]

        # Stack the list of tensors into a single tensor
        pil_images = torch.stack(pil_images)

        pil_images = pil_images.squeeze(1)
        if pil_images.size(0) < video_frames.size(0):
            last_element = pil_images[-1].unsqueeze(0)  # Shape [1, 3, 64, 64]

            # Repeat the last element to fill up the desired shape
            repeated_elements = last_element.repeat(video_frames.size(0)-pil_images.size(0), 1, 1, 1)  # Repeat 3 times

            # Concatenate the original tensor with the repeated elements
            pil_images = torch.cat((pil_images, repeated_elements), dim=0)  # Shape [25, 3, 64, 64]

        videos1.append(pil_images)
        videos2.append(video_frames)

    videos1 = torch.stack(videos1)
    videos2 = torch.stack(videos2)

    evaluator.empty_fake_stats()
    evaluator.compute_fake_stats(evaluator.load_videos(".pt", data_type="video_torch", video_data=videos2))
    score_fvd = evaluator.compute_fvd_from_stats()

    optic_flow = eval_optical_flow(videos1, videos2)
    wandb.log({"fvd": score_fvd})
    wandb.log({"optical flow": optic_flow})

    return score_fvd


def eval_optical_flow(real_videos, fake_videos):
    # # torch.Size([11, 25, 3, 64, 64]) torch.Size([11, 25, 3, 64, 64])
    model = Pips(stride=4).cuda()
    parameters = list(model.parameters())
    global_step = 0
    model.eval()

    global_step += 1
    N = 16 ** 2  # number of points to track

    total_distance1 = []
    with torch.no_grad():
        for video_idx in range(real_videos.size(0)):
            real_video = real_videos[video_idx].unsqueeze(0)
            fake_video = fake_videos[video_idx].unsqueeze(0)

            # print(trajs_e.size())  # torch.Size([1, 8, 256, 2])
            num_frames = 8
            real_video = real_video[:, ::3, :, :, :][:, :num_frames, :, :, :]  # Shape: [1, 8, 3, 360, 640]
            fake_video = fake_video[:, ::3, :, :, :][:, :num_frames, :, :, :]  # Shape: [1, 8, 3, 360, 640]

            trajs_real = run_model(model, real_video, N, None, None)  # torch.Size([1, 8, 3, 360, 640])
            trajs_fake = run_model(model, fake_video, N, None, None)

            for i in range(trajs_real.size(2)):  # 256 dimension (index 2)
                slice_real = trajs_real[0, :, i, :]
                slice_fake = trajs_fake[0, :, i, :]

                slice_real = slice_real.cpu().numpy()
                slice_fake = slice_fake.cpu().numpy()
                # indices = np.linspace(0, len(slice_real) - 1, num_frames, dtype=int)

                # slice_real = [slice_real[i] for i in indices]
                # slice_fake = [slice_fake[i] for i in indices]

                distance, path = fastdtw(slice_real, slice_fake, dist=euclidean)
                total_distance1.append(distance)

                 #distance2, path = fastdtw(slice_real, slice_fake2, dist=euclidean)
                # total_distance2.append(distance2)

    # fid_results = np.sum(fid_avg) / len(fid_avg)
    return np.sum(total_distance1) / N


def decode_latents(latents, vae, num_frames, decode_chunk_size=14):
    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
    # latents = latents.flatten(0, 1)

    latents = 1 / vae.config.scaling_factor * latents

    accepts_num_frames = "num_frames" in set(inspect.signature(vae.forward).parameters.keys())

    # decode decode_chunk_size frames at a time to avoid OOM
    frames = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        num_frames_in = latents[i: i + decode_chunk_size].shape[0]
        decode_kwargs = {}
        if accepts_num_frames:
            # we only pass num_frames_in if it's expected
            decode_kwargs["num_frames"] = num_frames_in

        frame = vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
        frames.append(frame)
    frames = torch.cat(frames, dim=0)

    # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
    frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    frames = frames.float()
    return frames


def main_eval(
        pretrained_model_path: str,
        validation_data: Dict,
        seed: Optional[int] = None,
        eval_file=None,
        **kwargs
):
    if seed is not None:
        set_seed(seed)
    # Load scheduler, tokenizer and models.
    pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(
        pretrained_model_path, eval=True)
    device = torch.device("cuda")
    pipeline.to(device)

    if eval_file is not None:
        eval_list = json.load(open(eval_file))
    else:
        eval_list = [[validation_data.prompt_image, validation_data.prompt]]

    output_dir = "output/svd_signal_v3_compact"
    iters = 5
    for example in eval_list:
        for t in range(iters):
            name, prompt = example
            out_file_dir = f"{output_dir}/{name.split('.')[0]}"
            os.makedirs(out_file_dir, exist_ok=True)
            out_file = f"{out_file_dir}/{t}.gif"
            validation_data.prompt_image = name
            validation_data.prompt = prompt
            eval(pipeline, vae_processor, validation_data, out_file, t)
            print("save file", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    cli_dict = OmegaConf.from_dotlist(args.rest)
    args_dict = OmegaConf.merge(args_dict, cli_dict)
    if args.eval:
        main_eval(**args_dict)
    else:
        main(**args_dict)
