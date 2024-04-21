import logging
from functools import wraps, partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch import sqrt
from torch.special import expm1
from tqdm.auto import tqdm

logg = logging.getLogger(__name__)


def exists(val):
    return val is not None


def identity(t):
    return t


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


def cast_tuple(t, l=1):
    return ((t,) * l) if not isinstance(t, tuple) else t


def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


def l2norm(t):
    return F.normalize(t, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift

    return inner


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner


class GaussianDiffusion:
    def __init__(
            self,
            pred_objective="eps",
            noise_schedule_f=logsnr_schedule_cosine,
            num_sample_steps=500,
            rescale_inputs=False,
            is_ddim=False,
            **kwargs,
    ):
        super().__init__()
        assert pred_objective in {
            "v",
            "eps",
        }, "whether to predict v-space (progressive distillation paper) or noise"
        logg.info(f"Using Gaussian diffusion with pred_objective={pred_objective}")
        self.model = None
        self.sampling_shape = None
        # training objective
        self.pred_objective = pred_objective
        # noise schedule
        self.log_snr = noise_schedule_f
        # sampling
        self.num_sample_steps = num_sample_steps
        self.eta = 0.0 if is_ddim else 1.0
        self.rescale_inputs = rescale_inputs
        self.is_ddim = is_ddim

    def get_alphas(self):
        device = "cuda" if exists(self.model) else "cpu"
        ts = torch.linspace(0, 1, self.num_sample_steps, device=device)
        log_snr = self.log_snr(ts)
        squared_alpha = log_snr.sigmoid()
        alpha = sqrt(squared_alpha)

        return alpha

    def log_alphas(self):
        device = self.model.device if exists(self.model) else "cpu"
        ts = torch.linspace(0, 1, self.num_sample_steps, device=device)
        log_snr = self.log_snr(ts)
        squared_alpha = log_snr.sigmoid()
        alpha = sqrt(squared_alpha)
        log_alpha = log(alpha)

        return log_alpha

    def get_betas(self):
        device = "cuda" if exists(self.model) else "cpu"
        ts = torch.linspace(0, 1, self.num_sample_steps, device=device)
        log_snr = self.log_snr(ts)
        squared_alpha = log_snr.sigmoid()
        alpha = sqrt(squared_alpha)
        beta = 1 - alpha

        return beta

    def set_fn(self, model):
        self.model = model

        # self.alphas = self.get_alphas()
        self.alphas_cumprod = self.get_alphas()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

    def set_sampling_shape(self, sampling_shape):
        self.sampling_shape = sampling_shape
        num_elements = np.prod(sampling_shape[1:])
        self.stroke_embedding_scale = np.sqrt(num_elements)

    def p_mean_variance(self, x, ctx, y, time, time_next, extra_cond=None):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
        pred, confidence = self.model(x, ctx, batch_log_snr, y, extra_cond=extra_cond)

        if self.pred_objective == "v":
            x_start = alpha * x - sigma * pred
        else:
            x_start = (x - sigma * pred) / alpha

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance, confidence

    # sampling related functions

    @torch.no_grad()
    def generalized_steps(self, ctx, y, presence, x=None, extra_cond=None):
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=ctx.device)
        x_t = x

        for i in tqdm(
                range(self.num_sample_steps),
                desc="sampling loop time step",
                total=self.num_sample_steps,
                disable=True,
        ):
            x_t = torch.where(presence.unsqueeze(-1), x_t, torch.zeros_like(x_t))
            times = steps[i]
            times_next = steps[i + 1]

            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
            squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

            alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

            batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
            pred, confidence = self.model(x, ctx, batch_log_snr, y, extra_cond=extra_cond)

            x0_t = (x_t - pred * (1 - alpha ** 2).sqrt()) / alpha
            c1 = (
                    self.eta * ((1 - alpha ** 2 / alpha_next ** 2) * (1 - alpha_next ** 2) / (1 - alpha ** 2)).sqrt()
            )
            c2 = ((1 - alpha_next ** 2) - c1 ** 2).sqrt()
            x_t = alpha_next * x0_t + c1 * torch.randn_like(x) + c2 * pred

            if torch.isnan(x_t).any():
                print("nan")
                print("alpha is nan", torch.isnan(alpha).any())
                print("alpha_next is nan", torch.isnan(alpha_next).any())
                print("pred is nan", torch.isnan(pred).any())
                print("x0_t is nan", torch.isnan(x0_t).any())
                print("c1 is nan", torch.isnan(c1).any())
                print("c2 is nan", torch.isnan(c2).any())
                exit()

        x_t = torch.where(presence.unsqueeze(-1), x_t, torch.zeros_like(x_t))

        return x_t, confidence

    @torch.no_grad()
    def p_sample(self, x, ctx, y, times, times_next, extra_cond=None):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance, confidence = self.p_mean_variance(
            x=x, ctx=ctx, y=y, time=times, time_next=times_next, extra_cond=extra_cond
        )

        if times_next == 0:
            return model_mean, confidence

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise, confidence

    @torch.no_grad()
    def p_sample_loop(self, ctx, y, presence, x=None, extra_cond=None):
        if self.rescale_inputs:
            ctx = ctx * self.stroke_embedding_scale

        if x is None:
            x = torch.randn(self.sampling_shape, device=ctx.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=ctx.device)

        for i in tqdm(
                range(self.num_sample_steps),
                desc="sampling loop time step",
                total=self.num_sample_steps,
                disable=True,
        ):
            x = torch.where(presence.unsqueeze(-1), x, torch.zeros_like(x))
            times = steps[i]
            times_next = steps[i + 1]
            x, confidence = self.p_sample(x, ctx, y, times, times_next, extra_cond=extra_cond)

        if self.rescale_inputs:
            x = x / self.stroke_embedding_scale

        return x, confidence

    @staticmethod
    def extract(a, t, x_shape):
        b, *_ = x_shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def predict_noise_from_start(self, x_t, t, x0):
        t = self.num_sample_steps - 1 - t
        return self.sqrt_recip_alphas_cumprod[t] * (x_t - x0) / self.sqrt_recipm1_alphas_cumprod[t]
        # return (
        #         (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
        #         self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        # )

    def model_predictions(
            self, x, ctx, y, time, alpha, sigma, clip_x_start=False, rederive_pred_noise=False, extra_cond=None, i=0,
    ):
        log_snr = self.log_snr(time)
        batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
        pred, confidence = self.model(x, ctx, batch_log_snr, y, extra_cond=extra_cond)
        maybe_clip = partial(torch.clamp, min=-2., max=2.) if clip_x_start else identity

        squared_alpha = log_snr.sigmoid()
        # model_output = self.model(x, t, x_self_cond)

        if self.pred_objective == "eps":
            pred_noise = pred
            x_start = (x - sigma * pred_noise) / alpha

            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, i, x_start)
            if torch.isnan(x_start).any():
                print("is x_start nan", torch.isnan(x_start).any())
                print("pred_noise", torch.isnan(pred_noise).any())
                print("alpha", torch.isnan(alpha).any())
                exit()
        elif self.pred_objective == "v":
            v = pred
            x_start = alpha * x - sigma * v
            pred_noise = sigma * x + alpha * v

        return pred_noise, x_start, confidence

    @torch.no_grad()
    def ddim_sample(self, ctx, y, presence, x=None, extra_cond=None):
        batch, device = ctx.shape[0], ctx.device

        if self.rescale_inputs:
            ctx = ctx * self.stroke_embedding_scale
        if x is None:
            x = torch.randn(self.sampling_shape, device=device)

        # times = torch.linspace(
        #     -1, self.num_sample_steps - 1, steps=self.num_sample_steps + 1
        # )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        # times = list(reversed(times.int().tolist()))
        # time_pairs = list(
        #     zip(times[:-1], times[1:])
        # )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=ctx.device)

        # times = torch.tensor(times, device=device, dtype=torch.long)

        for i in tqdm(
                range(self.num_sample_steps),
                desc="sampling loop time step",
                total=self.num_sample_steps,
                disable=True,
        ):
            time = steps[i]
            time_next = steps[i + 1]
            x = torch.where(presence.unsqueeze(-1), x, torch.zeros_like(x))

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            c = -expm1(log_snr - log_snr_next)

            squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
            squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

            alpha, sigma, alpha_next = map(
                sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
            )

            pred_noise, x_start, confidence = self.model_predictions(
                x, ctx, y, time, alpha, sigma, clip_x_start=True, rederive_pred_noise=True, extra_cond=extra_cond, i=i
            )

            if time_next < 0:
                x = x_start
                continue

            sigma = self.eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # check if x is nan
            if torch.isnan(x).any():
                print("NAN")
                print("alpha is nan?", torch.isnan(alpha).any())
                print("x_start is nan?", torch.isnan(x_start).any())
                print("pred_noise is nan?", torch.isnan(pred_noise).any())
                print("noise is nan?", torch.isnan(noise).any())
                print("alpha_next is nan?", torch.isnan(alpha_next).any())
                print("sigma is nan?", torch.isnan(sigma).any())
                print("c is nan?", torch.isnan(c).any())
                exit()

        if self.rescale_inputs:
            x = x / self.stroke_embedding_scale

        return x, confidence

    def available_samplers(self):
        # return [self.generalized_steps]
        return [self.p_sample_loop if not self.is_ddim else self.ddim_sample]
        # return [self.ddim_sample]

    # training related functions - noise prediction

    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, strokes, ctx, t, y, presence, noise=None, extra_cond=None):
        noise = default(noise, lambda: torch.randn_like(strokes))

        if self.rescale_inputs:
            strokes = strokes * self.stroke_embedding_scale
            ctx = ctx * self.stroke_embedding_scale

        x, log_snr = self.q_sample(x_start=strokes, times=t, noise=noise)
        x = torch.where(presence.unsqueeze(-1), x, torch.zeros_like(x))

        model_out, confidence = self.model(x, ctx, log_snr, y, extra_cond=extra_cond)

        if self.pred_objective == "v":
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * strokes

        elif self.pred_objective == "eps":
            target = noise

        loss = F.mse_loss(model_out, target, reduction="none")

        if presence is not None:
            loss = torch.where(presence.unsqueeze(-1), loss, torch.zeros_like(loss))
            divider = presence.sum()
        else:
            divider = loss.shape[0] * loss.shape[1] * loss.shape[2]

        nan_count = torch.sum(torch.isnan(loss))
        if nan_count > 0:
            logging.error(f"loss has {nan_count} nan elements")
            exit()
            # set nan elements to 0
            # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

        # position loss
        pos_loss = loss[:, :, 0:2]
        pos_loss = torch.sum(pos_loss) / divider / 2.0
        if torch.isnan(pos_loss):
            logging.warning("pos_loss is nan")
            pos_loss = torch.zeros_like(pos_loss).requires_grad_()

        # size loss
        size_loss = loss[:, :, 2:4]
        size_loss = torch.sum(size_loss) / divider / 2.0
        if torch.isnan(size_loss):
            logging.warning("size_loss is nan")
            size_loss = torch.zeros_like(size_loss).requires_grad_()

        # rotation loss
        rot_loss = loss[:, :, 4:5]
        rot_loss = torch.sum(rot_loss) / divider
        if torch.isnan(rot_loss) > 0:
            logging.warning("rot_loss is nan")
            rot_loss = torch.zeros_like(rot_loss).requires_grad_()

        # color loss
        color_loss = loss[:, :, 5:8]
        color_loss = torch.sum(color_loss) / divider / 3.0
        if torch.isnan(color_loss) > 0:
            logging.warning("color_loss is nan")
            color_loss = torch.zeros_like(color_loss).requires_grad_()

        loss_dict = {
            "pos_loss": pos_loss,
            "size_loss": size_loss,
            "rot_loss": rot_loss,
            "color_loss": color_loss,
        }

        return loss_dict, confidence

    def train_loss(self, strokes, ctx, y, presence=None, extra_cond=None):
        bs = strokes.shape[0]
        t = torch.zeros((bs,), device=ctx.device).float().uniform_(0, 1)

        return self.p_losses(strokes, ctx, t, y, presence, extra_cond=extra_cond)


# noise schedules


def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.0)


# converting gamma to alpha, sigma or logsnr


def gamma_to_alpha_sigma(gamma, scale=1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)


def gamma_to_log_snr(gamma, scale=1, eps=1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps=eps)


def safe_div(numer, denom, eps=1e-10):
    return numer / denom.clamp(min=eps)


if __name__ == "__main__":
    class MyLittleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(8, 8)

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(self, x, ctx, t, y):
            # print(f"x: {x.shape} ctx: {ctx.shape}, t: {t.shape}, y: {y.shape}")
            return self.model(x), None  # [:, :n // 2, :]


    model = MyLittleModel()
    sampling_shape = (2, 10, 8)

    diffusion = GaussianDiffusion()
    diffusion.set_fn(model)
    diffusion.set_sampling_shape(sampling_shape)

    data = torch.randn(2, 10, 8)
    ctx = torch.randn(2, 10, 8)
    y = torch.rand(
        2,
    )

    loss, _ = diffusion.train_loss(data, ctx, y)
    print(loss)

    img = diffusion.p_sample_loop(ctx, y)
    print(img.shape)
