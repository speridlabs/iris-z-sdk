import numpy as np
import torch
from einops import rearrange

from .model.sampling import get_schedule, scatter_ids

RGB = 0
DEPTH = 1
GIVEN_RGB = 2
GIVEN_DEPTH = 3
DEPTH_TIME_COORDINATE = 1


def grid_tokens(latent, time_coordinate):
    batch, _, height, width = latent.shape
    tokens = rearrange(latent, "b c h w -> b (h w) c")
    device = latent.device
    ids = torch.cartesian_prod(
        torch.tensor([time_coordinate], device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.tensor([0], device=device),
    )
    return tokens, ids.unsqueeze(0).expand(batch, -1, -1)


def rgbd_sequence(rgb_latent, depth_latent):
    """[RGB tokens, depth tokens] on the same (h, w) positions."""
    rgb_tokens, rgb_ids = grid_tokens(rgb_latent, 0)
    depth_tokens, depth_ids = grid_tokens(depth_latent, DEPTH_TIME_COORDINATE)
    tokens = torch.cat([rgb_tokens, depth_tokens], dim=1)
    ids = torch.cat([rgb_ids, depth_ids], dim=1)
    batch, rgb_length = rgb_tokens.shape[:2]
    modality = torch.cat(
        [
            torch.full((batch, rgb_length), RGB, device=tokens.device, dtype=torch.long),
            torch.full((batch, depth_tokens.shape[1]), DEPTH, device=tokens.device, dtype=torch.long),
        ],
        dim=1,
    )
    return tokens, ids, modality, rgb_length


def split_grids(tokens, ids, rgb_length):
    rgb_grid = torch.cat(scatter_ids(tokens[:, :rgb_length], ids[:, :rgb_length])).squeeze(2)
    depth_ids = ids[:, rgb_length:].clone()
    depth_ids[..., 0] = 0
    depth_grid = torch.cat(scatter_ids(tokens[:, rgb_length:], depth_ids)).squeeze(2)
    return rgb_grid, depth_grid


def guided_velocity(prediction, guidance):
    if prediction.shape[0] == 1:
        return prediction
    unconditional, conditional = prediction.chunk(2)
    return unconditional + guidance * (conditional - unconditional)


def initial_noise(shape, seed, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@torch.no_grad()
def sample_joint(model, context, height, width, seed, guidance, num_steps):
    ctx, ctx_ids = context
    device = ctx.device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, 128, height // 16, width // 16)
    rgb_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    depth_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    tokens, ids, modality, rgb_length = rgbd_sequence(rgb_noise, depth_noise)
    rows = ctx.shape[0]
    tokens = torch.cat([tokens] * rows)
    ids = torch.cat([ids] * rows)
    modality = torch.cat([modality] * rows)
    timesteps = get_schedule(num_steps, rgb_length)
    for t_current, t_previous in zip(timesteps[:-1], timesteps[1:]):
        t_vector = torch.full((rows,), t_current, dtype=dtype, device=device)
        prediction = model(
            x=tokens, x_ids=ids, timesteps=t_vector, ctx=ctx, ctx_ids=ctx_ids, guidance=None, modality_ids=modality
        )
        velocity = guided_velocity(prediction, guidance)
        tokens = tokens + (t_previous - t_current) * torch.cat([velocity] * rows)
    return split_grids(tokens[:1], ids[:1], rgb_length)


@torch.no_grad()
def sample_rgb(model, context, height, width, seed, guidance, num_steps):
    """The depth grid stays in the sequence as pure noise and is never denoised."""
    ctx, ctx_ids = context
    device = ctx.device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, 128, height // 16, width // 16)
    rgb_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    depth_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    tokens, ids, modality, rgb_length = rgbd_sequence(rgb_noise, depth_noise)
    rows = ctx.shape[0]
    batch_ids = torch.cat([ids] * rows)
    batch_modality = torch.cat([modality] * rows)
    depth_level = torch.ones((rows,), dtype=dtype, device=device)
    timesteps = get_schedule(num_steps, rgb_length)
    for t_current, t_previous in zip(timesteps[:-1], timesteps[1:]):
        t_vector = torch.full((rows,), t_current, dtype=dtype, device=device)
        prediction = model(
            x=torch.cat([tokens] * rows),
            x_ids=batch_ids,
            timesteps=t_vector,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
            modality_ids=batch_modality,
            rgb_timesteps=t_vector,
            depth_timesteps=depth_level,
            depth_start=rgb_length,
        )
        velocity = guided_velocity(prediction, guidance)
        tokens[:, :rgb_length] = tokens[:, :rgb_length] + (t_previous - t_current) * velocity[:, :rgb_length]
    return split_grids(tokens, ids, rgb_length)[0]


@torch.no_grad()
def sample_depth(model, context, rgb_latent, seeds, num_steps, average_last=0, decode=None):
    """One depth per seed. average_last = k > 0: the median of the last k clean-depth estimates, as log depth."""
    ctx, ctx_ids = context
    device = ctx.device
    dtype = next(model.parameters()).dtype
    batch = len(seeds)
    rgb_latent = rgb_latent.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
    clean, ids, modality, rgb_length = rgbd_sequence(rgb_latent, torch.zeros_like(rgb_latent))
    tokens = torch.cat([initial_noise((1,) + tuple(clean.shape[1:]), seed, device, dtype) for seed in seeds])
    tokens[:, :rgb_length] = clean[:, :rgb_length]
    modality[:, :rgb_length] = GIVEN_RGB
    ctx = ctx.expand(batch, -1, -1)
    ctx_ids = ctx_ids.expand(batch, -1, -1)
    rgb_level = torch.zeros((batch,), dtype=dtype, device=device)
    depth_ids = ids[:, rgb_length:].clone()
    depth_ids[..., 0] = 0
    timesteps = get_schedule(num_steps, rgb_length)
    first_averaged = len(timesteps) - 1 - average_last
    estimates = []
    for step, (t_current, t_previous) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vector = torch.full((batch,), t_current, dtype=dtype, device=device)
        velocity = model(
            x=tokens,
            x_ids=ids,
            timesteps=t_vector,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
            modality_ids=modality,
            rgb_timesteps=rgb_level,
            depth_timesteps=t_vector,
            depth_start=rgb_length,
        )
        if average_last and step >= first_averaged:
            estimate = tokens[:, rgb_length:] - t_current * velocity[:, rgb_length:]
            estimate_grid = torch.cat(scatter_ids(estimate, depth_ids)).squeeze(2)
            estimates.append([decode(estimate_grid[i : i + 1]) for i in range(batch)])
        tokens[:, rgb_length:] = tokens[:, rgb_length:] + (t_previous - t_current) * velocity[:, rgb_length:]
    if not average_last:
        depth_grid = split_grids(tokens, ids, rgb_length)[1]
        return [depth_grid[i : i + 1] for i in range(batch)]
    averaged = []
    for i in range(batch):
        members = [step_estimates[i] for step_estimates in estimates]
        aligned = [align_affine(member, members[-1]) for member in members]
        averaged.append(keep_sky(np.median(np.stack(aligned), axis=0), members))
    return averaged


@torch.no_grad()
def sample_rgb_from_depth(model, context, depth_latent, seed, guidance, num_steps):
    ctx, ctx_ids = context
    device = ctx.device
    dtype = next(model.parameters()).dtype
    rows = ctx.shape[0]
    depth_latent = depth_latent.to(device=device, dtype=dtype)
    clean, ids, modality, rgb_length = rgbd_sequence(torch.zeros_like(depth_latent), depth_latent)
    tokens = initial_noise(clean.shape, seed, device, dtype)
    tokens[:, rgb_length:] = clean[:, rgb_length:]
    modality[:, rgb_length:] = GIVEN_DEPTH
    batch_ids = torch.cat([ids] * rows)
    batch_modality = torch.cat([modality] * rows)
    depth_level = torch.zeros((rows,), dtype=dtype, device=device)
    timesteps = get_schedule(num_steps, rgb_length)
    for t_current, t_previous in zip(timesteps[:-1], timesteps[1:]):
        t_vector = torch.full((rows,), t_current, dtype=dtype, device=device)
        prediction = model(
            x=torch.cat([tokens] * rows),
            x_ids=batch_ids,
            timesteps=t_vector,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
            modality_ids=batch_modality,
            rgb_timesteps=t_vector,
            depth_timesteps=depth_level,
            depth_start=rgb_length,
        )
        velocity = guided_velocity(prediction, guidance)
        tokens[:, :rgb_length] = tokens[:, :rgb_length] + (t_previous - t_current) * velocity[:, :rgb_length]
    return split_grids(tokens, ids, rgb_length)[0]


SKY_VALUE = 1.0 - 1e-6


def align_affine(member, reference):
    design = np.stack([member.ravel(), np.ones(member.size)], axis=1)
    (scale, shift), _, _, _ = np.linalg.lstsq(design, reference.ravel(), rcond=None)
    return scale * member + shift


def keep_sky(averaged, members):
    """Pixels most members mark as sky (+1) stay at +1; alignment moves them slightly."""
    sky_votes = np.mean([member >= SKY_VALUE for member in members], axis=0)
    return np.where(sky_votes > 0.5, 1.0, averaged)


def ensemble(log_depths):
    reference = log_depths[0]
    aligned = [reference] + [align_affine(member, reference) for member in log_depths[1:]]
    return keep_sky(np.median(np.stack(aligned), axis=0), log_depths)
