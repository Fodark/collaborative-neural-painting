import gc
import json

# get the logger for this module
import logging
import os
from typing import Any, List, Optional

import math
import numpy as np
import torch
import torchvision.transforms as T
import wandb
from PIL import ImageDraw
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

from .components.snp.renderer import Renderer
from .components.utils import generate_context

log = logging.getLogger(__name__)


class MaskGitModule(LightningModule):
    def __init__(
        self,
        net,
        renderer: Renderer,
        lr: float,
        val_path: str,
        max_levels_length: Optional[List[int]] = None,
        scale: float = 2.0,
        batch_size: int = 64,
        max_tokens: int = 1024,
        compile_network: bool = False,
        T: int = 11,
        **kwargs,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["diffusion", "net", "renderer", "fid_score", "clip_score"]
        )
        self.hparams.lr = lr
        self.max_levels_length = max_levels_length
        self.eval_bs = batch_size
        self.scale = scale
        self.val_path = val_path
        self.compile_network = compile_network

        self.net: nn.Module = net
        self.renderer = renderer

        self.val_path: str = val_path
        self.fid_score = FrechetInceptionDistance(reset_real_features=False)

        self.real_strokes: Optional[torch.Tensor] = None
        self.class_: Optional[torch.Tensor] = None
        self.aug_params: Optional[torch.Tensor] = None
        self.lengths: Optional[torch.Tensor] = None
        self.levels_length: Optional[torch.Tensor] = None
        self.classes_names: Optional[List[str]] = None

        # self.infilled_images: List[List[torch.Tensor]] = [[], [], []]
        self.real_images = []
        self.ctx_images = []
        self.pred_images = []

        self.modes = ["block", "level", "square", "random", "unconditional"]

        self.max_tokens = max_tokens
        # self.sos_token = max_tokens + 1
        self.mask_token_id = max_tokens
        self.choice_temperature = 4.5
        self.T = T

        self.gamma = self.gamma_func("cosine")

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r**3
        else:
            raise

    def unnormalize(self, x):
        # log.info(f"max {x.max()}, min {x.min()}")
        x = x.float() / self.max_tokens
        x.clamp_(0, 1)
        return x

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

        for i in range(0, len(validation_images), self.eval_bs):
            batch = validation_images[i : i + self.eval_bs]
            batch = self.read_images(batch)
            batch = batch.to(self.device)
            self.fid_score.update(batch, real=True)

    @torch.no_grad()
    def update_fid_generated(self, generated_images):
        """
        Compute the FID stats for the generated images
        Args:
            generated_images: the generated images

        Returns:
        """
        for i in range(0, len(generated_images), self.eval_bs):
            batch = generated_images[i : i + self.eval_bs]
            batch = [(x * 255).to(torch.uint8) for x in batch]
            batch = torch.stack(batch)
            self.fid_score.update(batch.cuda(), real=False)

    def compute_fid(self, stage="val", mode="random"):
        fid_value = self.fid_score.compute()
        self.fid_score.reset()
        if self.trainer.is_global_zero:
            log.info(f"[Epoch: {self.current_epoch} | Mode: {mode}] FID: {fid_value:.3f}")
        # to_log = {"fid": fid_value}
        # wandb.log(to_log, step=self.global_step)
        # self.log(f"{stage}/fid", fid_value, sync_dist=True)
        return fid_value

    def log_fid(self):
        fid_value = self.fid_score.compute()
        log.info(f"FID: {fid_value:.3f}")
        to_log = {f"fid": fid_value}
        self.log_values("val", to_log)

    @staticmethod
    def infilling_losses(gt, prediction):
        gt = rearrange(gt, "(t c) -> t c", c=8)
        prediction = rearrange(prediction, "(t c) -> t c", c=8)

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

    def log_values(self, stage, metrics_dict):
        # update each key in metrics_dict prefixed with stage
        metrics_dict = {f"{stage}/{k}": v for k, v in metrics_dict.items()}
        metrics_dict["trainer/global_step"] = self.global_step
        self.logger.experiment.log(
            metrics_dict,
        )

    def add_class_to_images(self, images, classes):
        rendered = [o.clone() for o in images]
        transform = T.ToPILImage()
        downsample = T.Resize(128)
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
        strokes = rearrange(strokes, "b (t c) -> b t c", c=8)
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
        self._CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(self.device)
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
            self.trainer._data_connector._train_dataloader_source.dataloader().dataset.get_classes_names()
        )

    def forward(self, x, ctx, y, presence=None):
        return self.diffusion.train_loss(x, ctx, y, presence)

    def mask_sequence(self, x: torch.Tensor, mode=None, deterministic=False):
        r = math.floor(self.gamma(np.random.uniform()) * x.shape[1])
        # sample = torch.rand(x.shape, device=x.device).topk(r, dim=1).indices
        # mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        # mask.scatter_(dim=1, index=sample, value=True)
        *_, mask = generate_context(x, self.max_levels_length, mode=mode, deterministic=deterministic)

        masked_indices = self.mask_token_id * torch.ones_like(x, device=x.device)
        a_indices = ~mask * x + mask * masked_indices

        return a_indices, mask

    def step(self, batch: Any):
        x, y = batch["data"], batch["class_idx"]
        a_indices, _ = self.mask_sequence(x)

        # a_indices = torch.cat((sos_tokens, a_indices), dim=1)

        # target = torch.cat((sos_tokens, x), dim=1)
        target = x
        logits = self.net(a_indices, y)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss_dict = {"ce": loss}

        return loss_dict

    def create_input_tokens_normal(self, num):
        blank_tokens = torch.ones((num, self.num_image_tokens), device="cuda")
        masked_tokens = self.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        return masked_tokens.to(torch.int64)

    def tokens_to_logits(self, seq, class_):
        return self.net(seq, class_)

    @torch.no_grad()
    def sample_good(self, inputs=None, class_=None, num=1, mode="cosine"):
        # self.transformer.eval()
        # N = self.num_image_tokens
        T = self.T
        if inputs is None:
            inputs: torch.Tensor = self.create_input_tokens_normal(num)

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs  # [8, 257]
        for t in range(T):
            # print(f"t: {t}")
            logits = self.tokens_to_logits(
                cur_ids, class_
            )  # call transformer to get predictions [8, 257, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()

            unknown_map = (
                cur_ids == self.mask_token_id
            )  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(
                unknown_map, sampled_ids, cur_ids
            )  # replace all -1 with their samples and leave the others untouched [8, 257]

            ratio = 1.0 * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            selected_probs = torch.squeeze(
                torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1
            )  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]

            selected_probs = torch.where(
                unknown_map, selected_probs, self._CONFIDENCE_OF_KNOWN_TOKENS
            )  # ignore tokens which are already sampled [8, 257]

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning // 8 * mask_ratio), 1
            )  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(
                torch.zeros_like(mask_len),
                torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) // 8 - 1, mask_len),
            )  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(
                mask_len, selected_probs, temperature=self.choice_temperature * (1.0 - ratio)
            )
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)
        # return cur_ids[:, :]
        return cur_ids  # [:, :]

    @staticmethod
    def mask_by_random_topk(mask_len, probs, temperature=1.0):
        # print(f"mask_len.shape: {mask_len.shape}")
        # print(f"probs.shape: {probs.shape}")
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(
            0, 1
        ).sample(probs.shape).to("cuda")
        # print(f"confidence.shape: {confidence.shape}")
        confidence = rearrange(confidence, "b (n d) -> b n d", d=8)
        # print(f"confidence.shape: {confidence.shape}")
        confidence = torch.mean(confidence, dim=2)
        # print(f"confidence.shape: {confidence.shape}")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = confidence < cut_off
        # print(f"Masking shape: {masking.shape}")
        # print(f"Unmasked tokens: {torch.sum(masking == 0).item()}")
        # repeat each element 8 times
        masking = repeat(masking, "b n -> b (n d)", d=8)
        # print(f"Masking shape: {masking.shape}")
        return masking

    @staticmethod
    def mask_by_random_topk_strokes(mask_len, probs, temperature=1.0):
        print(f"mask_len.shape: {mask_len.shape}")
        print(f"probs.shape: {probs.shape}")

        # mask_len = mask_len // 8
        # take the mean prob every 8 elements
        probs = rearrange(probs, "b (n d) -> b n d", d=8)
        probs = torch.mean(probs, dim=1)
        # probs = probs.reshape(-1, 8, probs.shape[-1])

        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(
            0, 1
        ).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)

        # confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(
        #     0, 1
        # ).sample(probs.shape).to("cuda")
        # sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = confidence < cut_off

        print(f"Unmasked: {torch.sum(masking == False)}")
        print(f"masking.shape: {masking.shape}")

        # go back to the original shape, repeat the mask 8 times in dim 1
        masking = masking.repeat(1, 8)
        print(f"masking.shape: {masking.shape}")

        return masking

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        self.log_values("train", loss_dict)
        loss = sum(loss_dict.values())

        return {"loss": loss}

    def on_validation_start(self) -> None:
        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes))]

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
        self.log_image(f"val/real", self.real_images)
        for i, generated in enumerate(self.pred_images):
            ctx_with_class = self.add_class_to_images(
                self.ctx_images[i][-self.eval_bs :], self.class_[-self.eval_bs :]
            )
            self.log_image(f"val/ctx/{self.modes[i]}", [make_grid(ctx_with_class)])
            fake_with_class = self.add_class_to_images(
                generated[: self.eval_bs], self.class_[-self.eval_bs :]
            )
            self.log_image(f"val/fake/{self.modes[i]}", [make_grid(fake_with_class)])
            # self.log_image(
            #     f"val/ctx/{self.modes[i]}", [make_grid(self.ctx_images[i][: self.eval_bs])]
            # )
            # self.log_image(f"val/fake/{self.modes[i]}", [make_grid(generated[: self.eval_bs])])

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
        self.pred_images = [[] for _ in range(len(self.modes))]

        return metrics

    def on_test_start(self) -> None:
        self.on_fit_start()
        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes))]

    def test_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        self.log_values("test", loss_dict)
        loss = sum(loss_dict.values())

        # if batch_idx < 5:
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
        self.log_image(f"test/real", self.real_images[0:1])
        for i, generated in enumerate(self.pred_images):
            self.log_image(
                f"test/ctx/{self.modes[i]}", [make_grid(self.ctx_images[i][: self.eval_bs])]
            )
            self.log_image(f"test/fake/{self.modes[i]}", [make_grid(generated[: self.eval_bs])])

        # compute the mean of the metrics
        metrics = {}
        for metric in outputs[0].keys():
            metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()

        for idx, mode in enumerate(self.modes):
            self.update_fid_generated(self.pred_images[idx])
            fid_value = self.compute_fid("test", mode)
            metrics[f"fid/{mode}"] = fid_value

        import pprint

        pprint.pprint(metrics)

        self.log_values("test", metrics)

        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes))]

        return metrics

    def complete_with_method(self, gt, class_, idx, mode, stage):
        ctx, presence = self.mask_sequence(gt, mode)
        fake = self.sample_good(ctx, class_)
        # substitute the max token in ctx with 0
        ctx = ctx.masked_fill(presence, 0)
        unnormalized_gt = self.unnormalize(gt)
        unnormalized_ctx = self.unnormalize(ctx)
        unnormalized_fake = self.unnormalize(fake)

        unnormalized_gt = rearrange(unnormalized_gt, "b (s d) -> b s d", d=8)
        unnormalized_ctx = rearrange(unnormalized_ctx, "b (s d) -> b s d", d=8)
        unnormalized_fake = rearrange(unnormalized_fake, "b (s d) -> b s d", d=8)

        rendered_ctx = [
            self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for o in unnormalized_ctx
        ]
        self.ctx_images[idx].extend(rendered_ctx)
        # self.ctx_images.append(self.render_strokes(unnormalized_ctx, class_))

        ### INFILLING METRICS
        # keep track of the infilling metrics
        infilling_metrics = {}
        for f, r, p in zip(unnormalized_fake, unnormalized_gt, presence):
            p = rearrange(p, "(b d) -> b d", d=8)
            real_filtered = rearrange(r[p], "(b d) -> b d", d=8)
            fake_filtered = rearrange(f[p], "(b d) -> b d", d=8)
            infilling_metrics_sample = self.infilling_metrics_sample(real_filtered, fake_filtered)
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
        # self.log_values(stage, infilling_metrics)

        ### RENDERING METRICS
        # presence = rearrange(presence, "b (s d) -> b s d", d=8)
        # fake_normalized = torch.where(presence, unnormalized_fake, unnormalized_ctx)
        fake_normalized = unnormalized_fake.clamp(0, 1)

        rendered_fake = [
            self.renderer.draw_on_canvas(o.unsqueeze(0) + 1e-3).squeeze().detach().cpu()
            for o in fake_normalized
        ]
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
        # self.log_values(stage, image_metrics)

        metrics = {**infilling_metrics, **image_metrics}
        self.log_values(stage, metrics)

        return metrics

    def infill(self, stage):
        gc.collect()

        real, class_ = self.real_strokes, self.class_
        # x, ctx, presence = self.generate_context(real, self.max_levels_length, stage=stage)

        metrics = {}

        for idx, mode in enumerate(self.modes):
            if self.trainer.is_global_zero:
                log.info(f"({idx + 1}/{len(self.modes)}) [{stage}] infilling with mode: {mode}")
            metrics_mode = self.complete_with_method(real, class_, idx, mode, stage)
            # update the metrics
            metrics = {**metrics, **{k: metrics.get(k, 0) + v for k, v in metrics_mode.items()}}
            # metrics = {k: metrics.get(k, 0) + v for k, v in metrics_mode.items()}

        return metrics

    def infilling_metrics_sample(self, gt, fake):
        # gt and fake have shape [n, 8]
        # compute the cost matrix for hungarian algorithm, cost is defined as l1 distance on the first 4 dimensions
        C = torch.cdist(gt[:, :4].float(), fake[:, :4].float(), p=1).cpu().numpy()
        assert C.shape == (gt.shape[0], fake.shape[0])
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
