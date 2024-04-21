import gc
import json
# get the logger for this module
import logging
import os
import random
from random import choices
from typing import Any, List, Optional

import moviepy.editor as mpy
import numpy as np
import torch
import torchvision.transforms as T
import wandb
from PIL import ImageDraw
from einops import rearrange
from pytorch_lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

from .components.snp.renderer import Renderer

# from torchmetrics.multimodal.clip_score import CLIPScore

log = logging.getLogger(__name__)


class DiffusionModule(LightningModule):
    def __init__(
            self,
            net,
            diffusion,
            renderer: Renderer,
            lr: float,
            val_path: str,
            progressive: bool = False,
            max_levels_length: Optional[List[int]] = None,
            scale: float = 2.0,
            batch_size: int = 64,
            max_seq_len: int = 450,
            compile_network: bool = False,
            weight_choice: bool = False,
            unconditional_only: bool = False,
            condition_on_rendered_context: bool = False,
            number_of_patches: int = 4,
            cfg_enabled: bool = True,
            train_only_random: bool = False,
            robustness_eval: bool = False,
            **kwargs,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["diffusion", "net", "renderer", "fid_score", "clip_score"]
        )
        self.hparams.lr = lr
        self.progressive = progressive  # TODO: actually implement the false case of this
        self.max_levels_length = max_levels_length
        self.eval_bs = batch_size
        self.scale = scale
        self.val_path = val_path
        self.compile_network = compile_network
        self.weight_choice = weight_choice
        self.unconditional_only = unconditional_only
        self.condition_on_rendered_context = condition_on_rendered_context
        self.cfg_enabled = cfg_enabled
        self.train_only_random = train_only_random
        self.robustness_eval = robustness_eval

        self.net: nn.Module = net
        # log.info("Network compiled successfully")
        self.diffusion = diffusion
        self.diffusion.set_sampling_shape((self.eval_bs, max_seq_len, 8))
        self.diffusion.set_fn(self.net)
        self.renderer = renderer

        if self.condition_on_rendered_context:
            self.ctx_renderer = Renderer((128, 128), half_precision=True, morphology=False).eval().requires_grad_(False)
            self.net.condition_on_context()
            self.number_of_patches = number_of_patches

        self.val_path: str = val_path
        self.fid_score = FrechetInceptionDistance(reset_real_features=False)
        self.accuracy = BinaryAccuracy(multidim_average="global")
        # self.clip_score = CLIPScore().eval().requires_grad_(False)

        self.modes = ["block", "level", "square", "random", "unconditional"]
        if self.unconditional_only:
            self.modes = ["unconditional"]
        self.weights = [0.15, 0.15, 0.15, 0.15, 0.4]
        self.real_strokes: Optional[torch.Tensor] = None
        self.class_: Optional[torch.Tensor] = None
        self.aug_params: Optional[torch.Tensor] = None
        self.lengths: Optional[torch.Tensor] = None
        self.levels_length: Optional[torch.Tensor] = None
        self.classes_names: Optional[List[str]] = None

        # self.infilled_images: List[List[torch.Tensor]] = [[], [], []]
        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes) + 1)]

    def unnormalize(self, x):
        x = (x / self.scale) + 0.5
        x.clamp_(0, 1)
        return x

    def build_square(self, stroke_sequence, deterministic=False):
        # Get the number of length of the sequence in each batch
        seq_len, _ = stroke_sequence.shape
        # Pick a random stroke as the starting point
        starting_stroke_idx = torch.randint(0, seq_len, (1,)).item() if not deterministic else 0
        starting_stroke = stroke_sequence[starting_stroke_idx, :]
        # Define the size of the square
        size = 0.5 * self.scale
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

    def generate_context(self, x: torch.Tensor, max_levels_length, stage="train", mode=None, deterministic=False, p=None):
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
        batch_size, n, _ = x.shape
        device = x.device
        # available_modes = ["block", "level", "square", "random"]
        available_modes = self.modes if not self.train_only_random else ["random"]

        masks = []

        for _ in range(batch_size):
            mode = choices(available_modes, k=1, weights=self.weights if self.weight_choice else None)[
                0] if mode is None else mode
            if mode == "random":
                # pick a random mask probability in range [0.1, 0.9]
                if p is None:
                    mask_prob = torch.rand(1, device=device) * 0.7 + 0.2
                else:
                    mask_prob = torch.tensor(p, device=device)
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
            elif mode == "square":
                # pick the extremes points of a square big as most as 0.25
                mask = self.build_square(x[_], deterministic=deterministic)
                masks.append(mask)
            else:  # unconditional
                mask = torch.ones(n, device=device).bool()
                masks.append(mask)

        # FALSE IS WHERE CONTEXT IS
        ctx = x.clone()
        presence = torch.stack(masks)

        if stage == "train":
            # where mask is false, fill images with -min
            x = torch.where(presence[:, :, None], x, torch.zeros_like(x))
        else:
            # x = torch.where(mask[:, :, None], torch.randn_like(x), -self.scale / 2)
            x = torch.randn_like(x)
        # where mask is true, fill context with -1
        ctx = torch.where(presence[:, :, None], torch.zeros_like(ctx), ctx)
        extra_cond = None

        if self.condition_on_rendered_context:
            # with torch.no_grad():
            rendered_context = []
            n_to_render = ctx.shape[0]
            k_at_time = 32
            for i in range(0, n_to_render, k_at_time):
                rendered_context.append(
                    self.ctx_renderer.draw_on_canvas(
                        torch.where(presence[i: i + k_at_time, :, None], torch.zeros_like(ctx[i: i + k_at_time]),
                                    self.unnormalize(ctx[i: i + k_at_time]))
                    )
                )
            rendered_context = torch.cat(rendered_context, dim=0)
            # rendered_context = F.interpolate(rendered_context, size=(128, 128), mode="bilinear")
            extra_cond = rearrange(rendered_context, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h=self.number_of_patches,
                                   w=self.number_of_patches)

        return x, ctx, presence, extra_cond

    @staticmethod
    def read_images(paths):
        images = []
        for path in paths:
            images.append(read_image(path))
        images = torch.stack(images)
        return images

    def compute_fid_stats_validation_set(self):
        # iterate over the self.val_path in self.eval_bs chunks, load the images with read_images and update the self.evaluator real stats
        validation_images = os.listdir(self.val_path)
        validation_images = [os.path.join(self.val_path, x) for x in validation_images]

        for i in range(0, len(validation_images), self.eval_bs * 2):
            batch = validation_images[i: i + self.eval_bs * 2]
            batch = self.read_images(batch)
            batch = batch.to(self.device)
            self.fid_score.update(batch, real=True)
            del batch

    @torch.no_grad()
    def update_fid_generated(self, generated_images):
        """
        Compute the FID stats for the generated images
        Args:
            generated_images: the generated images

        Returns:
        """
        for i in range(0, len(generated_images), self.eval_bs):
            batch = generated_images[i: i + self.eval_bs]
            batch = [(x * 255).to(torch.uint8) for x in batch]
            batch = torch.stack(batch)
            self.fid_score.update(batch.cuda(), real=False)
            del batch

    def compute_fid(self, stage="val", mode="random"):
        fid_value = self.fid_score.compute()
        self.fid_score.reset()
        if self.trainer.is_global_zero:
            log.info(f"[Epoch: {self.current_epoch} | Mode: {mode}] FID: {fid_value:.3f}")
        # to_log = {"fid": fid_value}
        # wandb.log(to_log, step=self.global_step)
        # self.log(f"{stage}/fid", fid_value, sync_dist=True)
        return fid_value

    @torch.no_grad()
    def compute_and_log_clip_score(self, images, prompts):
        """
        Compute the CLIP score for a batch of images and prompts
        Args:
            images: the images
            prompts: the prompts

        Returns:
        """
        images = [(x * 255).to(torch.uint8) for x in images]
        # compute the CLIP score
        self.clip_score.update(images, prompts)
        # detach the clip score
        clip_score = self.clip_score.compute().detach()
        to_log = {"clip_score": clip_score}
        self.log_values("val", to_log)

        return

    # def hungarian_matching(self, gt, prediction):
    #     C = torch.cdist(gt[:4], prediction[:4], p=1)
    #     matching = Munkres().compute(C)

    @staticmethod
    def infilling_losses(gt, prediction):
        # perform hungarian matching on the gt and prediction based on the l1 loss of the 4 components
        # gt: [n, 8], prediction: [n, 8]
        # matching = hungarian_matching(gt, prediction)

        l1_losses = {
            "infilling/pos": F.l1_loss(gt[:, :2], prediction[:, :2]),
            "infilling/size": F.l1_loss(gt[:, 2:4], prediction[:, 2:4]),
            "infilling/rotation": F.l1_loss(gt[:, 4:5], prediction[:, 4:5]),
            "infilling/color": F.l1_loss(gt[:, 5:8], prediction[:, 5:8]),
            "infilling/mean": F.l1_loss(gt, prediction),
        }

        return l1_losses

    def log_image(self, key, images):
        self.logger.log_image(
            key,
            [wandb.Image(image) for image in images],
            step=self.global_step,
        )

    def log_values(self, stage: str, metrics_dict):
        # update each key in metrics_dict prefixed with stage
        metrics_dict = {f"{stage}/{k}": v for k, v in metrics_dict.items()}
        metrics_dict["trainer/global_step"] = self.global_step
        metrics_dict["epoch"] = self.current_epoch
        # pprint(metrics_dict)
        self.logger.experiment.log(
            metrics_dict,
            # step=self.global_step,
        )

    def add_class_to_images(self, images, classes):
        rendered = [o.clone() for o in images]
        transform = T.ToPILImage()
        downsample = T.Resize(128, antialias=True)
        reverse_t = T.ToTensor()
        # downsampled_images = [downsample(o) for o in rendered]
        rendered_pil = [transform(downsample(o)) for o in images]
        # for each image, write the corresponding class in the top left corner
        for i, img in enumerate(rendered_pil):
            class_idx = classes[i].item()
            class_name = self.classes_names[class_idx]
            ImageDraw.Draw(img).text(  # Image
                (0, 0),  # Coordinates
                f"class: {class_name}",  # Text
                (255, 255, 255),  # Color
            )
            rendered[i] = reverse_t(img)

        return rendered

    def render_strokes(self, strokes, class_=None):
        rendered = [
            self.renderer.draw_on_canvas(s.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for s in strokes
        ]
        if class_ is not None:
            rendered = self.add_class_to_images(rendered, class_)
        rendered_grid = make_grid(rendered, nrow=8)
        return rendered_grid

    def on_fit_start(self) -> None:
        """
        Called before training starts.
        Computes the FID stats for the validation set and compiles the network if self.compile_network is True
        Returns:

        """
        self.compute_fid_stats_validation_set()
        try:
            if self.compile_network:
                self.net = torch.compile(self.net)
                torch.set_float32_matmul_precision("high")
                log.info("Network compiled successfully")
        except Exception as e:
            log.error(f"Network compilation failed: {e}")
            exit(1)

        # get the classes names from the train dataloader
        self.classes_names = (
            (
                self.trainer._data_connector._train_dataloader_source.dataloader().dataset.get_classes_names()
            )
            if self.trainer._data_connector._train_dataloader_source is not None
            else (
                self.trainer._data_connector._test_dataloader_source.dataloader().dataset.get_classes_names()
            )
        )
        assert (
                self.classes_names is not None
        ), "Could not get the classes names from the dataloader"

        log.info(f"On fit start completed")

    def forward(self, x, ctx, y, presence=None, extra_cond=None):
        return self.diffusion.train_loss(x, ctx, y, presence, extra_cond)

    def step(self, batch: Any):
        x, y, levels_masks = batch["data"], batch["class_idx"], batch["levels_masks"]
        x, ctx, mask, extra_cond = self.generate_context(x, self.max_levels_length)

        loss_dict, confidence = self(x, ctx, y, mask, extra_cond=extra_cond)
        if confidence is not None:
            confidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                confidence.squeeze(-1), levels_masks.float()
            )
            loss_dict["confidence_loss"] = confidence_loss

        return loss_dict

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        log.info(f"Epoch {self.current_epoch} started")

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        self.log_values("train", loss_dict)
        loss = sum(loss_dict.values())

        return {"loss": loss}

    def on_validation_start(self) -> None:
        if self.cfg_enabled:
            self.diffusion.set_fn(self.net.forward_with_cfg)
        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes) + 1)]

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        self.log_values("val", loss_dict)
        loss = sum(loss_dict.values())

        if self.trainer.is_global_zero:
            log.info(f"[VAL] batch_idx: {batch_idx}, infilling")
        self.real_strokes, self.class_ = batch["data"], batch["class_idx"]
        real_s = self.unnormalize(self.real_strokes)
        self.real_s_rendered = self.render_strokes(real_s)
        self.real_images.append(self.real_s_rendered)
        metrics = self.infill("val")
        # put together the metrics
        metrics["val/loss"] = loss

        return metrics


    def validation_epoch_end(self, outputs: List[Any]):
        self.diffusion.set_fn(self.net)
        self.log_image(f"val/real", [self.real_images[0]])
        for i, _ in enumerate(self.modes):
            generated = self.pred_images[i]
            ctx_with_class = self.add_class_to_images(
                self.ctx_images[i][-self.eval_bs:], self.class_[-self.eval_bs:]
            )
            self.log_image(f"val/ctx/{self.modes[i]}", [make_grid(ctx_with_class)])
            fake_with_class = self.add_class_to_images(
                generated[-self.eval_bs:], self.class_[-self.eval_bs:]
            )
            self.log_image(f"val/fake/{self.modes[i]}", [make_grid(fake_with_class)])

        # compute the mean of the metrics
        metrics = {}
        for metric in outputs[0].keys():
            metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()
        if self.trainer.is_global_zero:
            log.info("Starting FID computation")
        for idx, mode in enumerate(self.modes):
            self.update_fid_generated(self.pred_images[idx])
            fid_value = self.compute_fid("val", mode)
            metrics[f"fid/{mode}"] = fid_value
        if self.trainer.is_global_zero:
            log.info("FID computation done")

        self.log_values("val", metrics)

        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes) + 1)]

        return metrics

    def on_test_start(self) -> None:
        self.on_fit_start()
        if self.cfg_enabled:
            self.diffusion.set_fn(self.net.forward_with_cfg)
        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes) + 1)]

    def test_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        loss = sum(loss_dict.values())

        if self.trainer.is_global_zero:
            log.info(f"[TEST] batch_idx: {batch_idx}, infilling")
        self.real_strokes, self.class_ = batch["data"], batch["class_idx"]
        real_s = self.unnormalize(self.real_strokes)
        self.real_s_rendered = self.render_strokes(real_s)
        self.real_images.append(self.real_s_rendered)
        metrics = self.infill("test")
        # put together the metrics
        metrics["test/loss"] = loss

        return metrics

    def test_epoch_end(self, outputs: List[Any]):
        self.diffusion.set_fn(self.net)
        self.log_image(f"test/real", [self.real_images[0]])

        modes = ["0.2", "0.4", "0.6", "0.8"] if self.robustness_eval else self.modes
        modes = ["unconditional"] if self.unconditional_only else modes

        for i, name in enumerate(modes):
            generated = self.pred_images[i]
            if i != len(self.modes):
                ctx_with_class = self.add_class_to_images(
                    self.ctx_images[i][-self.eval_bs:], self.class_[-self.eval_bs:]
                )
                self.log_image(f"test/ctx/{modes[i]}", [make_grid(ctx_with_class)])

            name = modes[i] if i != len(modes) else "progressive"
            fake_with_class = self.add_class_to_images(
                generated[-self.eval_bs:], self.class_[-self.eval_bs:]
            )
            self.log_image(f"test/fake/{name}", [make_grid(fake_with_class)])

        # compute the mean of the metrics
        metrics = {}
        for metric in outputs[0].keys():
            metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()

        for idx, mode in enumerate(modes):
            self.update_fid_generated(self.pred_images[idx])
            fid_value = self.compute_fid("test", mode)
            metrics[f"fid/{mode}"] = fid_value

        self.log_values("test", metrics)

        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes) + 1)]

        return metrics

    def infilling_metrics_sample(self, gt, fake):
        # gt and fake have shape [n, 8]
        # compute the cost matrix for hungarian algorithm, cost is defined as l1 distance on the first 4 dimensions
        C = torch.cdist(gt[:, :4].float(), fake[:, :4].float(), p=1).cpu().numpy()
        assert C.shape == (gt.shape[0], fake.shape[0])
        # check that no inf or -inf or nan are present
        min_inf = np.isneginf(C).any()
        max_inf = np.isposinf(C).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = C[~np.isinf(C)]
            min_values = values.min()
            max_values = values.max()
            m = min(C.shape)

            positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

            C[np.isinf(C)] = place_holder
        # compute the optimal assignment
        row_ind, col_ind = linear_sum_assignment(C)

        # reorder the fake strokes according to the optimal assignment
        fake = fake[col_ind]

        l1_losses = {
            "pos": F.l1_loss(gt[:, :2], fake[:, :2]),
            "size": F.l1_loss(gt[:, 2:4], fake[:, 2:4]),
            "rotation": F.l1_loss(gt[:, 4:5], fake[:, 4:5]),
            "color": F.l1_loss(gt[:, 5:8], fake[:, 5:8]),
            "mean": F.l1_loss(gt, fake),
        }

        return l1_losses

    def complete_with_method(self, gt, class_, idx, mode, stage, sampling_fn, p=None):
        x, ctx, presence, extra_cond = self.generate_context(
            gt, self.max_levels_length, stage=stage, mode=mode, p=p
        )
        fake, confidence = sampling_fn(ctx, class_, presence, x=x, extra_cond=extra_cond)

        metrics = {}

        fake.clamp_(-self.scale // 2, self.scale // 2)
        if confidence is not None:
            accuracy = self.accuracy(preds=confidence.squeeze(), target=presence.int())
            metrics[f"{mode}/accuracy"] = accuracy
            fake_full_unnormalized = self.unnormalize(fake)
            rendered_ff = [
                self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
                for o in fake_full_unnormalized
            ]
            self.log_image(f"{stage}/full/{mode}", [make_grid(rendered_ff)])

            confidence = confidence.sigmoid()
            # count elements falling under 0.5
            unconfident_elements = (confidence < 0.5).sum().item()
            if self.trainer.is_global_zero:
                log.info(
                    f"Using confidence, {unconfident_elements} elements are unconfident, ~{unconfident_elements // confidence.shape[0]} per element, accuracy: {accuracy * 100:.1f}%"
                )
            fake = torch.where(confidence > 0.5, fake, torch.ones_like(fake) * -self.scale // 2)

        unnormalized_fake = self.unnormalize(fake)
        unnormalized_gt = self.unnormalize(gt)
        unnormalized_ctx = self.unnormalize(ctx)
        unnormalized_ctx = torch.where(
            presence.unsqueeze(-1), torch.zeros_like(unnormalized_ctx), unnormalized_ctx
        )

        rendered_ctx = [
            self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for o in unnormalized_ctx
        ]
        self.ctx_images[idx].extend(rendered_ctx)

        ### INFILLING METRICS
        # keep track of the infilling metrics
        infilling_metrics = {}
        for f, r, p in zip(unnormalized_fake, unnormalized_gt, presence):
            infilling_metrics_sample = self.infilling_metrics_sample(r[p], f[p])
            # prepend the mode to the keys
            infilling_metrics_sample = {
                f"infilling/{mode}/{k}": v for k, v in infilling_metrics_sample.items()
            }
            # sum the metrics
            infilling_metrics = {
                k: infilling_metrics.get(k, 0) + v for k, v in infilling_metrics_sample.items()
            }
        # average the metrics
        infilling_metrics = {
            k: v / unnormalized_fake.shape[0] for k, v in infilling_metrics.items()
        }

        ### RENDERING METRICS
        fake_normalized = torch.where(presence[:, :, None], unnormalized_fake, unnormalized_ctx)
        fake_normalized.clamp_(0, 1)

        rendered_fake = [
            self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for o in fake_normalized
        ]

        if self.unconditional_only:
            outdir = "unconditional_only"
            os.makedirs(outdir, exist_ok=True)
            # save the rendered images
            for i, f in enumerate(rendered_fake):
                save_image(f, os.path.join(outdir, f"{idx}_{i}.png"))


        self.pred_images[idx].extend(rendered_fake)
        rendered_real = [
            self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for o in unnormalized_gt
        ]

        rendered_fake = torch.stack(rendered_fake)
        rendered_real = torch.stack(rendered_real)

        image_distance = F.mse_loss(rendered_fake, rendered_real)
        image_metrics = {
            f"image_l2/{mode}": image_distance,
        }

        metrics = {**infilling_metrics, **image_metrics, **metrics}

        return metrics

    def complete_progressive(self, gt, class_, idx, stage, sampling_fn):
        x, ctx, presence, extra_cond = self.generate_context(
            gt, self.max_levels_length, stage=stage, mode="unconditional"
        )

        final_strokes = torch.zeros_like(gt)

        for i in range(len(self.max_levels_length)):
            presence = torch.zeros_like(presence)
            sequence_start_curr_level = 0 if i == 0 else self.max_levels_length[i - 1]
            presence[:, sequence_start_curr_level:] = True
            fake, confidence = sampling_fn(ctx, class_, presence, x=x, extra_cond=extra_cond)
            # get the current level max length and remove the rest from fake
            # current_level_max_length = self.max_levels_length[i]
            # in final strokes, put the fake strokes of the current level
            final_strokes[:, sequence_start_curr_level: self.max_levels_length[i]] = fake[
                                                                                     :, sequence_start_curr_level:
                                                                                        self.max_levels_length[i]
                                                                                     ]
            final_strokes.clamp_(-self.scale // 2, self.scale // 2)
            ctx = final_strokes.clone()

        unnormalized_fake = self.unnormalize(final_strokes)
        unnormalized_fake.clamp_(0, 1)

        rendered_fake = [
            self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for o in unnormalized_fake
        ]
        self.pred_images[idx].extend(rendered_fake)

        return {}

    def infill(self, stage):
        gc.collect()

        real, class_ = self.real_strokes, self.class_
        sampling_fn = self.diffusion.available_samplers()[0]

        metrics = {}

        if self.robustness_eval:
            ps = [0.2, 0.4, 0.6, 0.8]
            for idx, p in enumerate(ps):
                if self.trainer.is_global_zero:
                    log.info(f"({idx + 1}/{len(ps)}) [{stage}] infilling with p: {p}")
                metrics_p = self.complete_with_method(real, class_, idx, "random", stage, sampling_fn, p)
                # update the metrics
                metrics = {**metrics, **{k: metrics.get(k, 0) + v for k, v in metrics_p.items()}}
            return metrics

        if self.unconditional_only:
            if self.trainer.is_global_zero:
                log.info(f"[{stage}] infilling with mode: unconditional")
            metrics_unconditional = self.complete_with_method(real, class_, 0, "unconditional", stage, sampling_fn)
            # update the metrics
            metrics = {**metrics, **{k: metrics.get(k, 0) + v for k, v in metrics_unconditional.items()}}
            return metrics

        for idx, mode in enumerate(self.modes):
            if self.trainer.is_global_zero:
                log.info(f"({idx + 1}/{len(self.modes)}) [{stage}] infilling with mode: {mode}")
            metrics_mode = self.complete_with_method(real, class_, idx, mode, stage, sampling_fn)
            # update the metrics
            metrics = {**metrics, **{k: metrics.get(k, 0) + v for k, v in metrics_mode.items()}}

        return metrics

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        opt = self.hparams.optimizer(params=self.net.parameters(), lr=self.hparams.lr)

        optimizer = [opt]
        scheduler = [{"scheduler": self.hparams.scheduler(optimizer=opt), "interval": "epoch"}]

        return optimizer, scheduler


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "autoregressive.yaml")
    _ = hydra.utils.instantiate(cfg)
