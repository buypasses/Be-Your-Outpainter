import os
import imageio
import numpy as np
from typing import Union
from pathlib import Path
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from einops import rearrange


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, save_png=True):
    """Save video frames as GIF and optionally as PNG sequence.

    Args:
        videos: Tensor of shape (b, c, t, h, w)
        path: Output path for GIF
        rescale: Whether to rescale from [-1,1] to [0,1]
        n_rows: Number of rows in grid
        fps: Frames per second for GIF
        save_png: If True, also saves individual frames as PNG sequence
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    print(videos.min())
    print(videos.max())
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=1000 * 1 / fps)

    # Also save as PNG sequence for better quality and reusability
    if save_png:
        png_dir = save_png_sequence(outputs, path)
        return path, png_dir
    return path, None


def save_png_sequence(frames: list, base_path: str) -> str:
    """Save frames as PNG sequence for lossless quality and easy manipulation.

    Args:
        frames: List of numpy arrays (H, W, C) in uint8 format
        base_path: Base path (e.g., 'output/sample.gif')

    Returns:
        Path to the PNG sequence directory
    """
    base_path = Path(base_path)
    png_dir = base_path.parent / f"{base_path.stem}_frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        frame_path = png_dir / f"frame_{i:06d}.png"
        Image.fromarray(frame).save(frame_path)

    print(f"Saved {len(frames)} PNG frames to {png_dir}")
    return str(png_dir)


def frames_to_video(png_dir: str, output_path: str, fps: int = 30, codec: str = "libx264"):
    """Convert PNG sequence to video using ffmpeg.

    Args:
        png_dir: Directory containing PNG frames (frame_000000.png, etc.)
        output_path: Output video path (.mp4, .webm, etc.)
        fps: Frames per second
        codec: Video codec (libx264 for mp4, libvpx-vp9 for webm with alpha)
    """
    import subprocess

    frame_pattern = str(Path(png_dir) / "frame_%06d.png")

    if output_path.endswith('.webm'):
        # WebM with alpha channel support
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
            output_path
        ]
    else:
        # Standard MP4
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', codec, '-pix_fmt', 'yuv420p',
            output_path
        ]

    subprocess.run(cmd, capture_output=True)
    print(f"Saved video to {output_path}")


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipeline.text_encoder(
        uncond_input.input_ids.to(pipeline.device)
    )[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
    ddim_scheduler,
):
    timestep, next_timestep = (
        min(
            timestep
            - ddim_scheduler.config.num_train_timesteps
            // ddim_scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        ddim_scheduler.alphas_cumprod[timestep]
        if timestep >= 0
        else ddim_scheduler.final_alpha_cumprod
    )
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet, clip_id, control):
    # noise_pred = unet(latents, t, clip_id=clip_id, encoder_hidden_states=context,control=control)["sample"]
    noise_pred = unet(latents, t, encoder_hidden_states=context, control=control)[
        "sample"
    ]
    return noise_pred


@torch.no_grad()
def ddim_loop(
    pipeline, ddim_scheduler, latent, num_inv_steps, prompt, clip_id, control
):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(
            latent, t, cond_embeddings, pipeline.unet, clip_id, control
        )
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(
    pipeline,
    ddim_scheduler,
    video_latent,
    num_inv_steps,
    prompt="",
    clip_id=None,
    control=None,
):
    ddim_latents = ddim_loop(
        pipeline,
        ddim_scheduler,
        video_latent,
        num_inv_steps,
        prompt,
        clip_id,
        control=control,
    )
    return ddim_latents


@torch.no_grad()
def ddim_inversion_long(
    pipeline,
    ddim_scheduler,
    video_latent,
    num_inv_steps,
    prompt="",
    window_size=16,
    stride=8,
    control=None,
    pixel_values=None,
    mask=None,
):
    if mask is not None:
        assert pixel_values is not None
        mask, masked_image = prepare_mask_and_masked_image(pixel_values, mask)
        bz, num_channels, video_length, height, width = video_latent.shape
        mask, masked_image_latents = pipeline.prepare_mask_latents(
            mask,
            masked_image,
            bz,
            height * pipeline.vae_scale_factor,
            width * pipeline.vae_scale_factor,
            video_latent.dtype,
            video_latent.device,
            None,
            False,
        )
        depth_map = rearrange(
            torch.cat([mask, masked_image_latents], dim=1),
            "(b f) c h w -> b c f h w",
            f=video_length,
        )
    elif pixel_values is not None and hasattr(pipeline, "prepare_depth_map"):
        video_length = video_latent.shape[2]
        depth_map = pipeline.prepare_depth_map(
            pixel_values,
            None,
            1,
            False,
            video_latent.dtype,
            video_latent.device,
        )
        depth_map = rearrange(depth_map, "(b f) c h w -> b c f h w", f=video_length)
    else:
        depth_map = None
    ddim_latents = ddim_loop_long(
        pipeline,
        ddim_scheduler,
        video_latent,
        num_inv_steps,
        prompt,
        window_size,
        stride,
        control,
        depth_map,
    )
    return ddim_latents


@torch.no_grad()
def ddim_loop_long(
    pipeline,
    ddim_scheduler,
    latent,
    num_inv_steps,
    prompt,
    window_size,
    stride,
    control,
    depth_map,
):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    video_length = latent.shape[2]
    views = get_views(video_length, window_size=window_size, stride=stride)
    count = torch.zeros_like(latent)
    value = torch.zeros_like(latent)
    for i in tqdm(range(num_inv_steps)):
        count.zero_()
        value.zero_()
        for t_start, t_end in views:
            control_tmp = None if control is None else control[:, :, t_start:t_end]
            latent_view = latent[:, :, t_start:t_end]
            t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
            if depth_map is not None:
                latent_input = torch.cat(
                    [latent_view, depth_map[:, :, t_start:t_end]], dim=1
                )
            else:
                latent_input = latent_view
            noise_pred = get_noise_pred_single(
                latent_input, t, cond_embeddings, pipeline.unet, t_start, control_tmp
            )
            latent_view_denoised = next_step(noise_pred, t, latent_view, ddim_scheduler)
            value[:, :, t_start:t_end] += latent_view_denoised
            count[:, :, t_start:t_end] += 1
        latent = torch.where(count > 0, value / count, value)
        all_latent.append(latent)
    return all_latent


def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start, t_end))
    return views
