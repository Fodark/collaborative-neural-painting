import argparse
import os
import traceback
import logging
from pathlib import Path
import pandas as pd

from decomposition.painter import Painter
import torch
from decomposition.utils import load_painter_config
from sorting.utils import StrokesLoader
import shutil
import glob
import pickle


def get_args():
    # settings
    parser = argparse.ArgumentParser(description="STYLIZED NEURAL PAINTING")
    parser.add_argument("--dataset_path", required=True, help="where the dataset will be stored")
    parser.add_argument(
        "--csv_file",
        default="/data/eperuzzo/brushstrokes-generation/code/dataset_acquisition/chunks/chunk_00.csv",
        type=str,
    )
    parser.add_argument(
        "--index_path",
        default="/data/eperuzzo/oxford_pet_params/oxford_pet_sorting_v2/",
        help="base folder with sorting results",
    )
    parser.add_argument(
        "--strokes_path",
        default="/data/eperuzzo/oxford_pet_params/oxford_pet_brushstrokes_params/",
        help="base folder with decomposition results",
    )
    parser.add_argument("--log_path", default="")
    parser.add_argument("--images_path", default="/data/eperuzzo/images/")
    parser.add_argument(
        "--painter_config",
        default="/data/eperuzzo/brushstrokes-generation/configs/decomposition/painter_config.yaml",
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU index")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # Create base directory
    Path(args.dataset_path).mkdir(parents=True, exist_ok=True)
    args.log_path = os.path.join(args.dataset_path, "logs")
    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    # Define Painter
    painter_config = load_painter_config(args.painter_config)
    painter_config.gpu_id = args.gpu_id  # overwrite
    pt = Painter(args=painter_config)
    df = pd.read_csv(args.csv_file)

    # Create directories and logging
    chunk = os.path.basename(args.csv_file).split(".")[0]
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.log_path, chunk + ".log"))
    logging.info(f"Total Number of images to process in this chunk: {len(df)}")

    # Update paths
    args.index_path = os.path.join(args.index_path, chunk)
    args.strokes_path = os.path.join(args.strokes_path, chunk)

    errors = []
    for index, row in df.iterrows():
        try:
            img_name = row["Images"]
            logging.info(f"Processing image {img_name}")

            tmp_path = os.path.join(args.dataset_path, img_name.split(".")[0])
            Path(tmp_path).mkdir(parents=True, exist_ok=True)

            # Copy stuff
            shutil.copy(
                src=os.path.join(args.images_path, img_name),
                dst=os.path.join(tmp_path, img_name),
            )

            shutil.copy(
                src=os.path.join(args.strokes_path, img_name.split(".")[0], "strokes_params.npz"),
                dst=os.path.join(tmp_path, "strokes_params.npz"),
            )

            # Render images with associate heuristic
            strokes_loader = StrokesLoader(path=tmp_path)
            strokes, layer = strokes_loader.load_strokes()

            idx_paths = glob.glob(
                os.path.join(args.index_path, img_name.split(".")[0], "lkh", "index", "*.pkl")
            )

            for idx_path in idx_paths:
                name = os.path.basename(idx_path)[:-4]  # .split('.')[0]
                os.mkdir(os.path.join(tmp_path, f"render_{name}"))

                with open(idx_path, "rb") as f:
                    idx = pickle.load(f)

                pt.inference(
                    strokes,
                    order=idx,
                    output_path=os.path.join(tmp_path, f"render_{name}"),
                    save_video=False,
                    save_jpgs=True,
                )

                shutil.copy(
                    src=os.path.join(idx_path),
                    dst=os.path.join(tmp_path, name + ".pkl"),
                )
        except Exception as e:
            logging.error(traceback.format_exc())
            errors.append(row["Images"])

    with open(os.path.join(args.log_path, f"errors_{chunk}.pkl"), "wb") as f:
        pickle.dump(errors, f)
