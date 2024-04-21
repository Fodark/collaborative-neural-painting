import gc
import json
# get the logger for this module
import logging
import os
from typing import Any, List, Optional

import torch
import torchvision.transforms as T
import wandb
from PIL import ImageDraw
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


class ContinuousModule(LightningModule):
    def __init__(
            self,
            net,
            renderer: Renderer,
            lr: float,
            val_path: str,
            max_levels_length: Optional[List[int]] = None,
            scale: float = 2.0,
            batch_size: int = 64,
            compile_network: bool = False,
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

        self.modes = ["block", "level", "square", "random", "unconditional"]
        self.real_strokes: Optional[torch.Tensor] = None
        self.class_: Optional[torch.Tensor] = None
        self.aug_params: Optional[torch.Tensor] = None
        self.lengths: Optional[torch.Tensor] = None
        self.levels_length: Optional[torch.Tensor] = None
        self.classes_names: Optional[List[str]] = None

        # self.infilled_images: List[List[torch.Tensor]] = [[], [], []]
        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes))]

    def unnormalize(self, x):
        x = (x / self.scale) + 0.5
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
            batch = validation_images[i: i + self.eval_bs]
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
            batch = generated_images[i: i + self.eval_bs]
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
            self.trainer._data_connector._train_dataloader_source.dataloader().dataset.get_classes_names()
        )

    def forward(self, x, ctx, y, presence=None):
        pass

    def step(self, batch: Any):
        x, y = batch["data"], batch["class_idx"]
        _, ctx, presence = generate_context(x, self.max_levels_length, self.scale)

        target = x
        pred, _ = self.net(ctx, y)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = torch.where(presence.unsqueeze(-1), loss, torch.zeros_like(loss))
        divider = presence.sum()
        loss = loss.sum() / divider
        loss_dict = {"mse": loss}

        return loss_dict

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

        # if batch_idx < 5:
        if self.trainer.is_global_zero:
            log.info(f"[VAL] batch_idx: {batch_idx}, infilling")
        self.real_strokes, self.class_ = batch["data"], batch["class_idx"]
        real_s = self.unnormalize(self.real_strokes)
        self.real_s_rendered = self.render_strokes(real_s)
        self.real_images.append(self.real_s_rendered)
        metrics = self.infill("val")
        # put together the metrics
        metrics["loss"] = loss

        return metrics

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_image(f"val/real", [self.real_images[0]])
        for i, generated in enumerate(self.pred_images):
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
        self.log_image(f"test/real", self.real_images[0])
        for i, generated in enumerate(self.pred_images):
            # self.log_image(
            #     f"test/ctx/{self.modes[i]}", [make_grid(self.ctx_images[i][: self.eval_bs])]
            # )
            ctx_with_class = self.add_class_to_images(
                self.ctx_images[i][-self.eval_bs:], self.class_[-self.eval_bs:]
            )
            self.log_image(f"test/ctx/{self.modes[i]}", [make_grid(ctx_with_class)])
            # self.log_image(f"test/fake/{self.modes[i]}", [make_grid(generated[: self.eval_bs])])
            fake_with_class = self.add_class_to_images(
                generated[-self.eval_bs:], self.class_[-self.eval_bs:]
            )
            self.log_image(f"test/fake/{self.modes[i]}", [make_grid(fake_with_class)])

        # compute the mean of the metrics
        metrics = {}
        for metric in outputs[0].keys():
            metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()

        for idx, mode in enumerate(self.modes):
            self.update_fid_generated(self.pred_images[idx])
            fid_value = self.compute_fid("test", mode)
            metrics[f"fid/{mode}"] = fid_value

        self.log_values("test", metrics)

        self.real_images = []
        self.ctx_images = [[] for _ in range(len(self.modes))]
        self.pred_images = [[] for _ in range(len(self.modes))]

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

    def complete_with_method(self, gt, class_, idx, mode, stage):
        _, ctx, presence = generate_context(
            gt, self.max_levels_length, stage=stage, scale=self.scale, mode=mode
        )
        fake, _ = self.net(ctx, class_)
        fake.clamp_(-self.scale // 2, self.scale // 2)
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
        # self.log_values(stage, infilling_metrics)

        ### RENDERING METRICS
        fake_normalized = torch.where(presence[:, :, None], unnormalized_fake, unnormalized_ctx)
        fake_normalized.clamp_(0, 1)

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
        # self.log_values(stage, metrics)

        return metrics

    def infill(self, stage):
        gc.collect()
        real, class_ = self.real_strokes, self.class_
        metrics = {}

        for idx, mode in enumerate(self.modes):
            if self.trainer.is_global_zero:
                log.info(f"({idx + 1}/{len(self.modes)}) [{stage}] infilling with mode: {mode}")
            metrics_mode = self.complete_with_method(real, class_, idx, mode, stage)
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


    def predict_dataloader(self):
        root_folder = "/data/shared/ndallasen/datasets/inp/awesome-animals-complete/predict/"
        data = []

        for idx, class_ in enumerate(os.listdir(root_folder)):
            for file in os.listdir(os.path.join(root_folder, class_)):
                file_path = os.path.join(root_folder, class_, file)
                strokes = torch.load(file_path, map_location="cpu").clamp(0, 1)
                # strokes = strokes * self.scale - self.scale // 2
                data.append( (strokes, torch.tensor(idx), file_path) )
                # print(data[-1])
        print(f"Loaded {len(data)} samples for prediction")
        return [data]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        ctx_out_root = "/data/shared/ndallasen/datasets/inp/awesome-animals-complete/user-study-ctx/"
        pred_out_root = "/data/shared/ndallasen/datasets/inp/awesome-animals-complete/user-study-pred-continuous-t/"
        os.makedirs(ctx_out_root, exist_ok=True)
        os.makedirs(pred_out_root, exist_ok=True)


        big_renderer = Renderer((720, 720), half_precision=False, morphology=True)
        classes_ = ['Rabbit', 'Horse', 'Bird', 'Eagle', 'Wolf', 'Dog', 'Duck', 'Tiger', 'Cat', 'Squirrel']
        metadata_path = os.path.join("/data/shared/ndallasen/datasets/inp", "metadata", "awesome-animals-complete.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        """Called for each batch in the dataloader."""
        # log.info(f"predict_step: {batch.shape}")
        # print(batch)
        # return
        strokes, class_idx, file_path = batch
        # file_path = "/data/shared/ndallasen/datasets/inp/awesome-animals-complete/val/Duck/seg_5694.pt"
        my_little_duck = torch.load(file_path,
                                    map_location="cpu")
        file_name: str = file_path.split("/")[-1]
        file_basename: str = file_name.split(".")[0]

        levels_length: List[int] = self.metadata[file_basename]
        # log.info(f"duck: {my_little_duck.shape}, levels: {levels_length}, max: {self.max_levels_length}")

        padded_levels = []

        for idx, max_levels_length_cum in enumerate(self.max_levels_length):
            # max_level_length = max_levels_length_cum - self.max_levels_length[
            #     idx - 1] if idx > 0 else max_levels_length_cum
            max_level_length = max_levels_length_cum
            current_level_length = levels_length[idx] - levels_length[idx - 1] if idx > 0 else levels_length[idx]
            # print(f"max_level_length: {max_level_length}, current_level_length: {current_level_length}")
            current_level_strokes = my_little_duck[(levels_length[idx - 1] if idx > 0 else 0):levels_length[idx]]
            if max_level_length > current_level_length:
                # log.info(f"max_levels_length: {max_level_length}, levels_length: {current_level_length}")
                if len(current_level_strokes.shape) == 1:
                    current_level_strokes = current_level_strokes.unsqueeze(0)
                padded_level = torch.cat(
                        [current_level_strokes, torch.zeros(
                            (max_level_length - current_level_length, 8))])
                padded_levels.append(padded_level)
            else:
                padded_levels.append(current_level_strokes)

        my_little_datapoint = torch.cat(padded_levels).unsqueeze(0).to(self.device).half()
        my_little_datapoint = (my_little_datapoint * self.scale) - self.scale // 2
        # class_idx = classes_.index("Duck")
        class_ = torch.tensor([class_idx]).unsqueeze(0).to(self.device)

        # sampling_fn = self.diffusion.available_samplers()[0]

        for mode_ in ["block", "random", "level", "square"]:
            _, ctx, presence = generate_context(
                my_little_datapoint, self.max_levels_length, stage="val", scale=self.scale, mode=mode_, deterministic=True
            )
            ctx = torch.where(presence[:, :, None], torch.ones_like(ctx) * -self.scale // 2, ctx)
            fake, _ = self.net(ctx, class_)
            fake.clamp_(-self.scale // 2, self.scale // 2)
            fake = self.unnormalize(fake)
            ctx = self.unnormalize(ctx)

            fake = torch.where(presence[:, :, None], fake, ctx)
            fake.clamp_(0, 1)


            # x, ctx, presence, extra_cond = self.generate_context(my_little_datapoint, self.max_levels_length, "val", mode_, deterministic=True)
            # fake, confidence = sampling_fn(ctx, class_, presence, x=x, extra_cond=extra_cond)
            # fake = self.unnormalize(fake)
            # fake = torch.where(presence[..., None], fake, self.unnormalize(ctx))
            rendered_fake = big_renderer.draw_on_canvas(fake + 1e-2, "black", False)[0]

            # ctx_for_animation = ctx.clone()
            # ctx_for_animation = self.unnormalize(ctx_for_animation)
            # ctx_for_animation = torch.where(presence[..., None], torch.zeros_like(ctx_for_animation, device=self.device), ctx_for_animation)
            # rendered_ctx = big_renderer.draw_on_canvas(ctx_for_animation + 1e-3, "black", False)[0]

            save_image(rendered_fake, os.path.join(pred_out_root, f"{file_basename}_{mode_}.png"))
            # save_image(rendered_ctx, os.path.join(ctx_out_root, f"{file_basename}_{mode_}.png"))


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "autoregressive.yaml")
    _ = hydra.utils.instantiate(cfg)
