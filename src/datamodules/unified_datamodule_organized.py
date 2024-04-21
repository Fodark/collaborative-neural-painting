import json
import os
from typing import Any, Dict, Optional, Tuple, List

import torch
from pytorch_lightning import LightningDataModule
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader, Dataset

from src import utils

log = utils.get_pylogger(__name__)


class MyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stage: str,
        max_levels_length: List[int],
        limit_data: Optional[int] = None,
        classes: Optional[List[str]] = None,
        scale=1,
    ):
        self.data_dir: str = data_dir
        self.data_list = []
        self.stage: str = stage
        self.scale = scale

        dataset_name = data_dir.split("/")[-1]
        root_dir = data_dir.split(dataset_name)[0]

        metadata_path = os.path.join(root_dir, "metadata", f"{dataset_name}.json")
        # assert os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}"
        if os.path.exists(metadata_path):
            # log.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                self.metadata: Dict[str, List[int]] = json.load(f)
        else:
            log.info(f"Metadata file not found at {metadata_path}")
            self.metadata = None

        self.max_levels_length = max_levels_length

        # self.limit_data = limit_data
        self.classes = classes
        self._indices = []
        self.classes_names = []

        # load the data
        self.load_data(stage)

    def load_data(self, stage):
        total_samples = 0
        # data_dir
        ## |--- train
        ## |--- |--- 0
        ## |--- |--- 1
        ## |--- val
        ## |--- |--- 0
        ## |--- |--- 1

        # folder structure is as above, with stage and inside the classes, inside the classes we have the .pt files
        stage_dir = os.path.join(self.data_dir, stage)
        for idx, class_name in enumerate(os.listdir(stage_dir)):
            self.classes_names.append(class_name)
            if self.classes is not None and class_name not in self.classes:
                continue
            class_dir = os.path.join(stage_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".pt"):
                    total_samples += 1
                    self.indices.append(idx)
                    self.data_list.append((os.path.join(class_dir, file), idx))

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, new_indices):
        self._indices = new_indices

    def get_classes_names(self):
        return self.classes_names

    def __len__(self) -> int:
        return len(self.data_list)

    def pad_data(
        self, data: torch.Tensor, levels_length: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad each layer the data to the max length, building a mask to tell real strokes from padding
        Args:
            data: the original complete sequence
            levels_length: the original length of each level in data

        Returns:
            padded_data: the padded data, concatenated in a single tensor
            masks: the masks to tell real strokes from padding in a single tensor
        """

        padded_data = []
        masks = []
        for i, level_length in enumerate(levels_length):
            if i >= len(self.max_levels_length):
                break
            start = 0 if i == 0 else levels_length[i - 1]
            data_current_level = data[start:level_length]
            if len(data_current_level.shape) == 1:
                data_current_level.unsqueeze_(0)
            mask_current_level = torch.ones(self.max_levels_length[i], dtype=torch.bool)
            mask_current_level[data_current_level.shape[0] :] = False

            try:
                if data_current_level.shape[0] < self.max_levels_length[i]:
                    data_current_level = torch.cat(
                        (
                            data_current_level,
                            torch.zeros(
                                self.max_levels_length[i] - data_current_level.shape[0],
                                data_current_level.shape[1],
                                device=data_current_level.device,
                            ),
                        )
                    )
            except Exception as e:
                print(e)
                print(data_current_level.shape)
                print(self.max_levels_length[i])
                exit(1)
            padded_data.append(data_current_level)
            masks.append(mask_current_level)

        return torch.cat(padded_data), torch.cat(masks)

    def __getitem__(self, idx: int) -> Dict:
        file_path, class_idx = self.data_list[idx]

        # get the file name
        file_name: str = file_path.split("/")[-1]
        file_basename: str = file_name.split(".")[0]

        if self.metadata is not None:
            levels_length: List[int] = self.metadata[file_basename]

        class_idx = torch.tensor([class_idx])

        with open(file_path, "rb") as f:
            data: torch.Tensor = torch.load(f, map_location="cpu").float()

        data.clamp_(0, 1).float()
        # data = data * 2 - 1.
        if self.metadata is not None:
            padded_data, masks = self.pad_data(data, levels_length)
        else:
            # pad the data to the max length
            padded_data = torch.cat(
                [
                    data,
                    torch.zeros(
                        self.max_levels_length[0] - data.shape[0],
                        data.shape[1],
                        device=data.device,
                    ),
                ],
                dim=0,
            )
            masks = torch.ones(self.max_levels_length[0], dtype=torch.bool)
        # padded_data, masks = self.pad_data(data, levels_length)
        padded_data = (padded_data - 0.5) * self.scale

        # log the range of padded data
        # log.info(f"padded_data: {padded_data.min()} - {padded_data.max()}")

        return {
            "data": padded_data,
            "levels_masks": masks,
            "class_idx": class_idx,
        }


class StrokesDatamodule(LightningDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[MyDataset] = None
        self.data_val: Optional[MyDataset] = None
        self.data_test: Optional[MyDataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        data_dir = os.path.join(self.hparams.datasets_root, self.hparams.dataset)
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MyDataset(
                data_dir=data_dir,
                stage="train",
                max_levels_length=self.hparams.max_levels_length,
                limit_data=self.hparams.limit_data,
                classes=self.hparams.classes,
                scale=self.hparams.scale,
            )
            self.data_val = MyDataset(
                data_dir=data_dir,
                stage="val",
                max_levels_length=self.hparams.max_levels_length,
                classes=self.hparams.classes,
                scale=self.hparams.scale,
            )
            self.data_test = MyDataset(
                data_dir=data_dir,
                stage="test",
                max_levels_length=self.hparams.max_levels_length,
                classes=self.hparams.classes,
                scale=self.hparams.scale,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        mpcs = MPerClassSampler(
            self.data_val.indices,
            m=2,
        )
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
            sampler=mpcs,
        )

    def test_dataloader(self):
        mpcs = MPerClassSampler(
            self.data_val.indices,
            m=2,
            # batch_size=self.hparams.batch_size
        )
        return DataLoader(
            # dataset=self.data_val,
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
            sampler=mpcs,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "strokes.yaml")
    # cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
