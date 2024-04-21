import os
from pathlib import Path
import argparse
import traceback
import logging

from decomposition import utils
from decomposition.painter import Painter

import pickle
from datetime import datetime
import pandas as pd
import numpy as np


def get_args():
    # settings
    parser = argparse.ArgumentParser(description="STROKES DECOMPOSITION")
    parser.add_argument("--output_path", required=True, type=str, help="Output path")
    parser.add_argument("--csv_file", required=True, type=str, help="Images to process")
    parser.add_argument("--data_path", required=True, type=str, help="where images are stored")
    parser.add_argument("--painter_config", default="../configs/decomposition/painter_config.yaml")
    parser.add_argument("--plot_loss", default=False)
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU index")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # Define Painter
    painter_config = utils.load_painter_config(args.painter_config)
    painter_config.gpu_id = args.gpu_id  # overwrite
    pt = Painter(args=painter_config)
    df = pd.read_csv(args.csv_file)

    # Create directories and logging
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    chunk_name = os.path.basename(args.csv_file).split(".")[0]
    logging.basicConfig(
        level=logging.INFO, filename=os.path.join(args.output_path, chunk_name + ".log")
    )
    logging.info(f"Total Number of images to process in this chunk: {len(df)}")

    errors = []
    for index, row in df.iterrows():
        try:
            start = datetime.now()
            logging.info("Processing image: {}, {}/{}".format(row["Images"], index, len(df)))
            img_name = row["Images"]

            img_path = os.path.join(args.data_path, img_name)
            tmp_output_path = os.path.join(args.output_path, chunk_name, img_name.split(".")[0])
            Path(tmp_output_path).mkdir(parents=True, exist_ok=True)
            # --------------------------------------------------------------------------------------------------------------
            # Decomposition
            painter_config.img_path = img_path
            strokes = pt.train()
            pt._save_stroke_params(strokes, path=tmp_output_path)
            final_img, alphas = pt.inference(strokes)
            np.savez_compressed(os.path.join(tmp_output_path, "alpha.npz"), alpha=alphas)
            # --------------------------------------------------------------------------------------------------------------
            # Save loss curves dictionary and figures
            elapsed = datetime.now() - start
            logs = pt.loss_dict
            logs["elapsed_time"] = elapsed
            with open(os.path.join(tmp_output_path, "logs.pkl"), "wb") as f:
                pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
            if args.plot_loss:
                utils.plot_loss_curves(logs, tmp_output_path)
        except Exception as e:
            logging.error(traceback.format_exc())
            errors.append(row["Images"])

    with open(os.path.join(args.output_path, f"errors_{chunk_name}.pkl"), "wb") as f:
        pickle.dump(errors, f)
