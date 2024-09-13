from typing import Callable, Dict, List, Optional, Union
import torch
from einops import rearrange, repeat
import PIL

from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid, \
    TextToVideoSDPipelineOutput
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import tensor2vid as svd_tensor2vid, \
    StableVideoDiffusionPipelineOutput
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor

from models.layerdiffuse_VAE import LatentSignalEncoder
import torch.nn.functional as F
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# from models.unet_3d_condition import UNet3DConditionModel
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def load_channel(channels, frame_step, frame_range_indices):
    if frame_range_indices[0] - frame_step >= 0:
        partial_channels = channels[frame_range_indices[0] - frame_step:frame_range_indices[-1], :]

    else:
        partial_channels = []

        for i in range(0, len(frame_range_indices)):
            # frame_range_indices[i] 2
            # frame_range_indices[i-1 ] 0
            if i == 0:
                if frame_range_indices[i] >= frame_step:
                    tmp_channel = channels[frame_range_indices[i] - frame_step:frame_range_indices[i], :]
                    assert tmp_channel.size(0) == frame_step
                    partial_channels.append(tmp_channel)
                else:
                    tmp_channel = channels[0:frame_range_indices[i], :]

                    first_element = channels[0]
                    first_element = first_element.unsqueeze(0)
                    # print("here", frame_step, frame_range_indices[i], first_element.size())
                    first_element_duplicated = first_element.repeat(frame_step - frame_range_indices[i], 1)
                    tmp_channel = torch.cat((first_element_duplicated, tmp_channel), dim=0)
                    assert tmp_channel.size(0) == frame_step, tmp_channel.size()
                    partial_channels.append(tmp_channel)
            else:
                frame_diff = frame_range_indices[i] + 1 - frame_range_indices[i - 1]

                # frame_diff 3
                if frame_diff == frame_step + 1:
                    tmp_channel = channels[frame_range_indices[i - 1]:frame_range_indices[i], :]
                    assert tmp_channel.size(0) == frame_step
                    partial_channels.append(tmp_channel)
                else:
                    tmp_channel = channels[frame_range_indices[i - 1]:frame_range_indices[i], :]

                    first_element = channels[frame_range_indices[i - 1]]
                    first_element = first_element.unsqueeze(0)
                    first_element_duplicated = first_element.repeat(frame_step - frame_diff + 1, 1)
                    tmp_channel = torch.cat((first_element_duplicated, tmp_channel), dim=0)
                    assert tmp_channel.size(0) == frame_step, tmp_channel.size()
                    partial_channels.append(tmp_channel)

            partial_channels = torch.cat(partial_channels, dim=0)
    return partial_channels


class MaskStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            image_encoder: CLIPVisionModelWithProjection,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(vae, image_encoder, unet, scheduler, feature_extractor)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _get_add_time_ids(
            self,
            fps,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
            self,
            batch_size,
            num_frames,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
            self,
            video,
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 25,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 3.0,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: int = 0.02,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
            n_input_frames=5,
            mask=None,
            signal=None,
            sig1=None,
            sig2=None,
            img1=None
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            video (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(video, height, width)

        # 2. Define call parameters
        if isinstance(video, PIL.Image.Image):
            batch_size = 1
        elif isinstance(video, list):
            batch_size = len(video)
        else:
            batch_size = 1

        # assert batch_size == 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0
        # print(video.size()) # should be torch.Size([3, 480, 640]
        # video = video.half().to(device)
        # 3. Encode input image
        video = video.unsqueeze(0).to(device)
        dtype = self.vae.dtype

        # enocde image latent
        image = video[:, 0:n_input_frames].to(dtype)
        image = rearrange(image, 'b f c h w-> (b f) c h w').to(dtype)

        # print("image", image.size())  # torch.Size([2, 3, 512, 512]) -> # torch.Size([(2, 5), 3, 512, 512])
        import math, random
        noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
        image = image + noise_aug_strength * torch.randn_like(image)
        image_latent = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor
        image_latent = rearrange(image_latent, '(b f) c h w-> (b c) f h w', b=batch_size).to(dtype)

        # print("image_latent", image_latent.size())  # torch.Size([2, 4, 64, 64]) -> # torch.Size([(2, 5), 4, 64, 64])
        # image_latent = rearrange(image_latent, '(b f) c h w-> b f c h w', b=bsz).to(dtype)
        image_pool = img1.to(image_latent.device)
        image_latent = image_pool(image_latent)
        print("pool", image_latent.size())

        image_latent = rearrange(image_latent, '(b c) f h w-> b c f h w', b=batch_size).to(dtype)
        image_latent = torch.squeeze(image_latent, dim=2)

        # image_embeddings = self._encode_image(video[0], device, num_videos_per_prompt, do_classifier_free_guidance)
        print("image_latent", image_latent.size()) # image_latent torch.Size([1, 4, 60, 80])

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        # image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        image_latent = repeat(image_latent, 'b c h w->b f c h w', f=num_frames)

        # mask = repeat(mask, '1 h w -> 2 f 1 h w', f=num_frames)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            video.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_latent.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale
        # signal
        signal_values = signal.float().half()  # [FPS, 512]

        signal_values = torch.nan_to_num(signal_values, nan=0.0)
        target_fps = 25

        max_frame = signal_values.size(0)

        native_fps = 20
        sample_fps = fps + 1
        frame_step = max(1, round(native_fps / sample_fps))
        frame_range = range(0, max_frame, frame_step)

        frame_range_indices = list(frame_range)[1:1 + num_frames]

        # print(signal_values.unsqueeze(0).size())
        signal_values = load_channel(signal_values, frame_step, frame_range_indices)
        signal_values = signal_values.unsqueeze(0)

        signal_encoder = sig1
        signal_encoder2 = sig2
        # bsz = signal_values.size(0)
        # fps = signal_values.size(1)

        # [B, FPS, 512] -> [B * FPS, 512]
        signal_values_reshaped = rearrange(signal_values, 'b (f c) h-> b f c h', c=frame_step)  # [B, FPS, 32]
        signal_values_reshaped_input = signal_values_reshaped[:, 0:n_input_frames]
        # print("signal_values_reshaped_input", signal_values_reshaped.size())

        signal_embeddings = signal_encoder(signal_values_reshaped_input)
        # print("signal_encoder", signal_values_reshaped.size())

        signal_embeddings = signal_embeddings.reshape(batch_size, 1, -1)
        signal_embeddings2 = signal_encoder2(signal_values_reshaped)

        # encoder_hidden_states = torch.cat((image_embeddings, signal_embeddings), dim=2)
        encoder_hidden_states = signal_embeddings.to(dtype)
        mask = signal_embeddings2.to(dtype)
        # print("mask ", mask.size()) # mask, torch.Size([1, 25, 1, 56, 72])

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        image_latent = torch.cat([image_latent] * 2) if do_classifier_free_guidance else image_latent

        encoder_hidden_states = torch.cat(
            [encoder_hidden_states] * 2) if do_classifier_free_guidance else encoder_hidden_states

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # k.size(), latent_model_input.size(), image_latent.size())
                # torch.Size([2, 25, 1, 8, 8]) torch.Size([2, 25, 4, 8, 8]) torch.Size([2, 25, 4, 8, 8])
                # torch.Size([2, 20, 1, 8, 8]) torch.Size([2, 20, 4, 8, 8]) torch.Size([2, 20, 4, 8, 8])

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([mask, latent_model_input, image_latent], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = svd_tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


class LatentToVideoPipeline(TextToVideoSDPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt=None,
            height=None,
            width=None,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale=9.0,
            negative_prompt=None,
            eta: float = 0.0,
            generator=None,
            latents=None,
            output_type="np",
            return_dict: bool = True,
            callback=None,
            callback_steps: int = 1,
            signal=None,
            sig1=None,
            sig2=None,
            sig3=None,
            cross_attention_kwargs=None,
            condition_latent=None,
            mask=None,
            timesteps=None,
            motion=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,
                usually at the expense of lower video quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            signal_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, signal, None
        )

        batch_size = 1
        # device = self._execution_device
        device = latents.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input signal
        # signal [FPS, 512]
        signal_values = signal.float().half()  # [FPS, 512]

        signal_values = torch.nan_to_num(signal_values, nan=0.0)
        target_fps = 25

        native_fps = 20
        sample_fps = num_frames
        frame_step = max(1, round(native_fps / sample_fps))

        frame_range = range(0, signal_values.size(0), frame_step)
        frame_range_indices = list(frame_range)[:target_fps]

        signal_values = signal_values[frame_range_indices, :]

        signal_values = signal_values.unsqueeze(0)
        # print("unsqueeze", signal_values.size())

        # signal_encoder = LatentSignalEncoder(input_dim=signal_values.size(-1) * signal_values.size(-2), output_dim=1024).to(device)
        # signal_encoder2 = LatentSignalEncoder(output_dim=latents.size(-1) * latents.size(-2)).to(device)
        signal_encoder = sig1.to(latents.device)
        signal_encoder2 = sig2.to(latents.device)
        signal_encoder3 = sig3.to(latents.device)

        bsz = signal_values.size(0)
        fps = signal_values.size(1)

        # Encode text embeddings
        # token_ids = batch['prompt_ids']
        # signal_encoder = LatentSignalEncoder(output_dim=1024).to(latents.device)

        signal_values_resized = rearrange(signal_values, 'b f c-> b (f c)')
        # print(signal_values.size())
        signal_embeddings = signal_encoder(signal_values_resized)
        signal_embeddings = signal_embeddings.reshape(bsz, 1, -1)

        signal_embeddings2 = signal_encoder2(signal_values).half()
        # print("signal_embeddings2", signal_embeddings2.size())

        signal_embeddings2 = rearrange(signal_embeddings2, 'b f (c h w)-> (b f) c h w', c=1,
                                       h=100, w=100)  # [B, FPS, 32]
        # print("after signal_embeddings2", signal_embeddings2.size())
        # signal_values torch.Size([2, 25, 512])
        # signal_embeddings2 torch.Size([2, 25, 10000])
        # after signal_embeddings2 torch.Size([2, 25, 1, 100, 100])
        signal_embeddings2 = F.interpolate(signal_embeddings2, size=(latents.size(-2), latents.size(-1)),
                                           mode='bilinear')
        signal_embeddings2 = rearrange(signal_embeddings2, '(b f) c h w-> b c f h w', b=bsz)  # [B, FPS, 32]

        # mask = batch["mask"]
        # mask = mask.div(255).to(dtype)

        # noisy_latents:  torch.Size([8, 4, 20, 64, 64])
        # mask = rearrange(mask, 'b h w -> b 1 1 h w')
        # mask = repeat(mask, 'b 1 1 h w -> (t b) 1 f h w', t=sample.shape[0] // mask.shape[0], f=sample.shape[2])
        # noisy_latents = torch.cat([mask, noisy_latents], dim=1)
        # freeze = repeat(condition_latent, 'b c 1 h w -> b c f h w', f=video_length)

        # torch.Size([8, 1, 20, 64, 64]) torch.Size([8, 4096])
        # print(signal_embeddings2.size(), signal_embeddings3.size())

        # encoder_hidden_states = torch.cat((image_embeddings, signal_embeddings), dim=2)
        encoder_hidden_states = signal_embeddings
        uncond_hidden_states = torch.zeros_like(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states] * 2) if do_classifier_free_guidance else encoder_hidden_states

        # mask torch.Size([2, 25, 1, 64, 64]) torch.Size([2, 25, 1, 64, 64])
        # mask = torch.cat((signal_embeddings2, signal_embeddings3), dim=2)
        # signal_embeddings2 -> [b, 1, f, h, w]
        signal_embeddings2 = F.pad(signal_embeddings2, (0, 0, 0, 0, 0, 1))
        mask = signal_embeddings2

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if timesteps is None:
            timesteps = self.scheduler.timesteps
        else:
            num_inference_steps = len(timesteps)
        # 5. Prepare latent variables. do nothing

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        uncondition_latent = condition_latent
        condition_latent = torch.cat(
            [uncondition_latent, condition_latent]) if do_classifier_free_guidance else condition_latent
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if motion is not None:
                    motion = torch.tensor(motion, device=device)
                # print(latent_model_input.size(), encoder_hidden_states.size())
                # torch.Size([2, 4, 20, 55, 74]) torch.Size([154, 1, 1024])
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,  # TODO: to be replaced
                    cross_attention_kwargs=cross_attention_kwargs,
                    condition_latent=condition_latent,
                    mask=mask,
                    motion=motion
                ).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video, latents

        return TextToVideoSDPipelineOutput(frames=video)


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]
