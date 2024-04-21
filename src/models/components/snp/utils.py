import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch.nn.functional as F


def img2patches(img: torch.Tensor, m_grid: int, s: int):
    """
    Divide the images in m_grid ** 2 patches of size s. The input is resized before dividing in patches

    :param img: [bs, c, H, W]
    :param m_grid: number of blocks
    :param s: size of patch
    :return: [bs x L, c, s, s] L = number of blocks
    """
    img = torch.nn.functional.interpolate(img, (m_grid * s, m_grid * s))
    bs, c, img_size, _ = img.shape
    patches = torch.nn.functional.unfold(
        img, kernel_size=(s, s), stride=s
    )  # [bs, c*s*s, m_grid * m_grid]
    patches = patches.permute(0, 2, 1).reshape(-1, c, s, s)  # [bs * m_gird * m_grid, c, s, s]
    return patches


def patches2img(patches: torch.Tensor, m_grid: int):
    """
    Recombine patches back to image.

    :param patches: [bs * L, c, s, s]
    :param m_grid: number of blocks
    :return: [bs, c, img_size, img_size]
    """
    _, c, s, _ = patches.shape
    patches = patches.reshape(-1, m_grid * m_grid, c * s * s).permute(0, 2, 1)  # [bs, c*s*s, L]
    img = torch.nn.functional.fold(
        patches, output_size=(s * m_grid, s * m_grid), kernel_size=(s, s), stride=s
    )  # [bs, c, img_size, img_size]
    return img


def make_even(x: int):
    """
    Check if it is even, otherwise add 1
    :param x:
    :return:
    """
    if x % 2 != 0:
        x += 1
    return x


def sample_uniform(r_min: float, r_max: float, size, device):
    return (r_min - r_max) * torch.rand(size, device=device) + r_max


def save_torch_img(img: torch.Tensor, path: str):
    """
    :param x: [..., 3, h, w] torch image, leading dimension can be batch size or channels
    :param path: path where the image is stored
    :return:
    """
    if len(img.shape) == 4:
        img = img[0]
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    plt.imsave(path, img)


def compute_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr


# ======================================================================================================================
# Renderer
def read_img(img_path, img_type="RGB", h=None, w=None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    return img


class Erosion2d(nn.Module):
    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode="constant", value=1e9)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.min(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel

        return x


class Dilation2d(nn.Module):
    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode="constant", value=-1e9)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.max(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel

        return x
