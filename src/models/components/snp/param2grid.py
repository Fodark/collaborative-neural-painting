import torch
import math
import numpy as np
import pdb


@torch.jit.script
def tensor_linspace(start: torch.Tensor, end: torch.Tensor, steps: int):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def param2grid(strokes: torch.Tensor, freq: int, margin: float):
    """
    Given the parameters of the strokes, returns a grid of points inside the stroke
    :param
    : strokes: [N x 8]
    :freq:
    :margin: how much we can move inside the stroke
    return: [N x freq ** 2], for each input a sequence of freq ** 2 points inside
    """
    xc, yc, w, h, theta = strokes[:, :5].T

    # apply margin
    w = (1 - margin) * w
    h = (1 - margin) * h

    # compute angle
    if freq > 1:
        alpha_w = torch.where(w >= h, math.pi * (theta - 0.5), math.pi * theta)[:, None]
        alpha_h = torch.where(w >= h, math.pi * theta, math.pi * (theta - 0.5))[:, None]

        xh = tensor_linspace(-0.5 * h, 0.5 * h, freq) * torch.cos(alpha_h)
        yh = tensor_linspace(-0.5 * h, 0.5 * h, freq) * torch.sin(alpha_h)
        xw = tensor_linspace(-0.5 * w, 0.5 * w, freq) * torch.cos(alpha_w)
        yw = tensor_linspace(-0.5 * w, 0.5 * w, freq) * torch.sin(alpha_w)

        # Sum points
        X = (
            xc[:, None].repeat(1, freq * freq)
            + torch.repeat_interleave(xh, freq, -1)
            + xw.repeat(1, freq)
        )
        Y = (
            yc[:, None].repeat(1, freq * freq)
            + torch.repeat_interleave(yh, freq, -1)
            + yw.repeat(1, freq)
        )
    else:
        X = xc
        Y = yc

    pts = torch.stack([X, Y], dim=-1)
    pts = torch.reshape(pts, (-1, freq, freq, 2))
    return pts  # N x freq x freq x 2
