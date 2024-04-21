import random
from random import choices

import torch
from einops import rearrange


def build_square(stroke_sequence, scale, deterministic=False):
    # Get the number of length of the sequence in each batch
    seq_len, _ = stroke_sequence.shape
    # Pick a random stroke as the starting point
    starting_stroke_idx = torch.randint(0, seq_len, (1,)).item() if not deterministic else 0
    starting_stroke = stroke_sequence[starting_stroke_idx, :]
    # Define the size of the square
    size = 0.5 * scale
    # Extract the x, y coordinates of the starting stroke
    x = starting_stroke[0]
    y = starting_stroke[1]

    # Define the square centered around the starting stroke
    left = x - size / 2
    right = x + size / 2
    top = y + size / 2
    bottom = y - size / 2

    # create a boolean tensor indicating if the strokes fall inside the square
    inside_square = (
            (stroke_sequence[:, 0] >= left)
            & (stroke_sequence[:, 0] <= right)
            & (stroke_sequence[:, 1] >= bottom)
            & (stroke_sequence[:, 1] <= top)
    )

    return inside_square


def generate_context(x: torch.Tensor, max_levels_length, scale=1.0, stage="train", mode=None, deterministic=False):
    """
    Given the noised input and the clean input, generate the context for the network,
    which is part of th sequence without noise to guide the network to generate the
    prediction for the noisy part of the sequence.
    Args:
        x: the noised input at given timestep
        max_levels_length: the maximum length of each level in the sequence
        mode: the mode to generate the context, can be "random", "future" or "level"
        stage: the stage of the training, can be "train" or "val"

    Returns:
        noised_input: the noised input at given timestep with strokes of context removed
        context: the context to guide the network, strokes to be predicted are removed
        mask: the mask to tell noise from context
    """
    should_flatten_at_end = x.ndim == 2
    if should_flatten_at_end:
        x = rearrange(x, "b (t c) -> b t c", c=8) / 127.5 - 1
    batch_size, n, _ = x.shape
    device = x.device
    available_modes = ["block", "level", "square", "random", "unconditional"]

    masks = []

    for _ in range(batch_size):
        mode = choices(available_modes, k=1)[0] if mode is None else mode
        if mode == "random":
            # pick a random mask probability in range [0.1, 0.9]
            mask_prob = torch.rand(1, device=device) * 0.7 + 0.2
            # generate a boolean mask of shape [batch_size, n]
            mask = torch.rand(n, device=device) < mask_prob
            masks.append(mask)
        elif mode == "level":
            # pick a random number in [0, 1, 2]
            max_n_levels = len(max_levels_length) - 1
            chosen_level = torch.randint(1, max_n_levels, (1,), device=device).item() if not deterministic else 1
            mask = torch.ones(n, device=device).bool()
            # for each row, set the last future_quantity[i] to false

            # get the length of the current level
            if chosen_level != 0:
                # context is up to max_levels_length[level[i]]
                mask[: max_levels_length[chosen_level]] = False
            masks.append(mask)
        elif mode == "block":
            # for each element in the batch pick a random block long 25% of the sequence starting from a random position
            # initialize mask with all false
            mask = torch.zeros(n, device=device).bool()
            # pick the length of the block
            length = random.randint(10, 3 * n // 4) if not deterministic else 60
            # pick the starting position of the block
            start = torch.randint(0, n - length - 1, (1,), device=device).item() if not deterministic else 100

            # set the block to true
            mask[start: start + length] = True
            masks.append(mask)
        elif mode == "square":  # mode == "square"
            # pick the extremes points of a square big as most as 0.25
            mask = build_square(x[_], scale, deterministic=deterministic)
            masks.append(mask)
        else:  # unconditional
            mask = torch.ones(n, device=device).bool()
            masks.append(mask)

    # FALSE IS WHERE CONTEXT IS
    ctx = x.clone()
    mask = torch.stack(masks)

    if stage == "train":
        # where mask is false, fill images with -min
        x = torch.where(mask[:, :, None], x, -scale / 2)
    else:
        x = torch.randn_like(x)
    # where mask is true, fill context with -1
    ctx = torch.where(mask[:, :, None], torch.zeros_like(ctx), ctx)

    if should_flatten_at_end:
        x = rearrange(x, "b t c -> b (t c)")
        ctx = rearrange(ctx, "b t c -> b (t c)")
        # mask has shape [batch_size, n], make it [batch_size, n, 8] and flatten it
        mask = rearrange(mask, "b t -> b t ()") * torch.ones(8, device=device, dtype=torch.bool)
        mask = rearrange(mask, "b t c -> b (t c)")

    return x, ctx, mask
