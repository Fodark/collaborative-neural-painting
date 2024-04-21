import glob
import os

import PIL.Image as Image
from typing import List, Tuple
from renderer import Renderer

import torch
import torch.optim as optim
from utils import img2patches, patches2img, make_even, sample_uniform, compute_psnr
import torchvision.transforms.functional as TF
import pickle as pkl


class OilPainterBase:
    """
    Modified from original Painter class. It contains only oil-painting related code.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Net G
        # as in SNP, keep output size 128 during parameter's optimization
        self.renderer = Renderer(canvas_size=[128, 128])
        self.output_size = 128
        self.d_shape = 5  # (x,y,w,h,theta)
        self.d_color = 3  # (r, g, b)
        self.d = 8  # total number of params per stroke

        # define some other vars to record the training states
        self.x = None

        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.G_final_pred_canvas = None

        self.G_loss = torch.tensor(0.0)
        self.step_id = 0
        self.anchor_id = 0
        self.output_dir = args.output_dir
        self.lr = args.lr

        # define the loss functions
        self._pxl_loss = torch.nn.L1Loss(reduction="none")
        # self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=False) not used

        # some other vars to be initialized in child classes
        self.input_aspect_ratio = None
        self.img_batch = None
        self.img = None
        self.final_rendered_images = None
        self.m_grid = None
        self.m_strokes_per_block = None

    def _compute_acc(self):

        target = self.img_batch.detach()
        canvas = self.G_pred_canvas.detach()
        psnr = compute_psnr(canvas, target, PIXEL_MAX=1.0)

        return psnr

    @staticmethod
    def save_stroke_params(v: torch.Tensor, output_path: str):
        """
        :param v: [bs, L, n_params] strokes to save
        :param output_path: bs paths where store each result
        :return:
        """
        with open(output_path, "wb") as handle:
            pkl.dump(v.cpu().numpy().astype("float32"), handle, protocol=pkl.HIGHEST_PROTOCOL)

    def _normalize_strokes_and_reshape(self, inp):
        v = inp.clone().detach()
        v = torch.reshape(
            v,
            (
                self.batch_size,
                self.m_grid * self.m_grid,
                self.m_strokes_per_block,
                self.d,
            ),
        )

        # === Create bias tensor
        idx = torch.arange(self.m_grid, device=v.device)
        y_ids = torch.repeat_interleave(idx, self.m_grid)
        x_ids = idx.repeat(self.m_grid)

        # === Normalize coordinates add bias
        v[:, :, :, 0] = (x_ids[None, :, None] + v[:, :, :, 0]) / self.m_grid
        v[:, :, :, 1] = (y_ids[None, :, None] + v[:, :, :, 1]) / self.m_grid
        v[:, :, :, 2] /= self.m_grid
        v[:, :, :, 3] /= self.m_grid

        # === Reshape to remove the L dimension
        v = torch.reshape(v, (self.batch_size, -1, self.d))
        return v

    def initialize_params(self):

        N = self.batch_size * self.m_grid * self.m_grid
        params_shape = (N, self.m_strokes_per_block, self.d)

        # Shape
        self.x = torch.randn(params_shape, dtype=torch.float32).to(self.device)

    def stroke_sampler(self, anchor_id: int):
        """
        :param anchor_id: int, current index of the stroke to be added to currently optimized strokes
        :return:
        """
        if anchor_id == self.m_strokes_per_block:
            return

        # Compute error map, blur and raise to power
        err_maps = torch.sum(
            torch.abs(self.img_batch - self.G_final_pred_canvas), dim=1, keepdim=True
        ).detach()
        ks = int(self.output_size / 8) + 1
        err_maps = TF.gaussian_blur(err_maps, (ks, ks))
        err_maps = torch.pow(err_maps, 4)
        err_maps = torch.flatten(err_maps, start_dim=1)  # remove channel, flatten last two dim
        err_maps[err_maps < 0] = 0

        for i in range(err_maps.shape[0]):
            if torch.all(err_maps[i] == 0):
                err_maps[i] = torch.ones_like(err_maps[i])
        err_maps = err_maps / (torch.sum(err_maps, dim=-1, keepdim=True) + 1e-99)

        # === Select stroke parameters based on error map ===
        index = torch.multinomial(err_maps, 1)  # Sample for each element based on the error
        y = torch.div(index, self.output_size, rounding_mode="floor") / self.output_size
        x = torch.remainder(index, self.output_size) / self.output_size
        center = torch.cat((x, y), dim=-1)

        # Sample color and remove extra dimensions
        color = torch.nn.functional.grid_sample(
            self.img_batch, grid=2 * center[:, None, None, :] - 1, align_corners=False
        )
        color = color.flatten(start_dim=1)  # remove the two extra dimension

        # Sample w,h,theta
        wh = sample_uniform(
            r_min=0.1,
            r_max=0.25,
            size=(self.m_grid * self.m_grid * self.batch_size, 2),
            device=self.device,
        )
        theta = sample_uniform(
            r_min=0,
            r_max=1,
            size=(self.m_grid * self.m_grid * self.batch_size, 1),
            device=self.device,
        )

        # === Assign sampled strokes ===
        self.x.data[:, anchor_id, :] = torch.cat((center, wh, theta, color), dim=-1)

    def _backward_x(self):
        self.G_loss = 0
        pixel_loss = self._pxl_loss(self.G_final_pred_canvas, self.img_batch)
        self.G_loss += self.args.beta_L1 * torch.mean(pixel_loss)
        self.G_loss.backward()

    def _forward_pass(self):

        v = self.x[:, 0 : self.anchor_id + 1, :]  # [bs x L, m_strokes_per_block, 8]
        v = v.reshape(-1, 8)  # [-1, n_params], flatten first dim

        (
            self.G_pred_foregrounds,
            self.G_pred_alphas,
        ) = self.renderer.render_single_strokes(v)
        ch_alpha = self.G_pred_alphas.size(1)  # can be either 1 or 3 according to the renderer

        # [bs, L, ch, H, W]
        self.G_pred_foregrounds = self.G_pred_foregrounds.reshape(
            -1, self.anchor_id + 1, 3, self.output_size, self.output_size
        )
        self.G_pred_alphas = self.G_pred_alphas.reshape(
            -1, self.anchor_id + 1, ch_alpha, self.output_size, self.output_size
        )

        for i in range(self.anchor_id + 1):
            G_pred_foreground = self.G_pred_foregrounds[:, i]
            G_pred_alpha = self.G_pred_alphas[:, i]
            self.G_pred_canvas = G_pred_foreground * G_pred_alpha + self.G_pred_canvas * (
                1 - G_pred_alpha
            )

        self.G_final_pred_canvas = self.G_pred_canvas


# ======================================================================================================================
class ProgressiveOilPainter(OilPainterBase):
    """
    Perform progressive parameters search, dividing the canvas to finer grid.
    """

    def __init__(self, args):
        super(ProgressiveOilPainter, self).__init__(args=args)

        self.batch_size = None
        self.x = None
        self.args = args

        self.max_divide = args.max_divide
        self.max_m_strokes = args.max_m_strokes

        self.m_strokes_per_block = self.stroke_parser()
        self.m_grid = 1

    def stroke_parser(self):

        total_blocks = 0
        for i in range(0, self.max_divide + 1):
            total_blocks += i**2

        return int(self.max_m_strokes / total_blocks)

    def load_img(self, paths: List) -> torch.Tensor:
        """
        :param paths: paths of the images to be loaded
        :return: tensor with the images resized and batched
        """
        batch = []
        self.aspect_ratio = []
        target_size = self.max_divide * self.output_size
        for path in paths:
            img = Image.open(path).convert("RGB")
            w, h = img.size

            # Pad the shorted side
            leading_dim = max(w, h)  # find the longest side
            scale = target_size / leading_dim

            h_new = make_even(int(scale * h))
            w_new = make_even(int(scale * w))

            # Compute padding
            h_pad = int((target_size - h_new) / 2)
            w_pad = int((target_size - w_new) / 2)

            # Resize and Pad
            img = TF.to_tensor(img)
            img = TF.resize(img, (h_new, w_new))
            img = TF.pad(img, (w_pad, h_pad))
            batch.append(img)

        batch = torch.stack(batch)
        self.batch_size = batch.size(0)
        return batch

    def compute_accuracy(self, verbose=True):
        acc = self._compute_acc().item()
        if verbose:
            print(
                "iteration step %d, G_loss: %.5f, step_acc: %.5f, grid_scale: %d / %d, strokes: %d / %d"
                % (
                    self.step_id,
                    self.G_loss.item(),
                    acc,
                    self.m_grid,
                    self.max_divide,
                    self.anchor_id + 1,
                    self.m_strokes_per_block,
                )
            )

    def clamp(self, val: float):
        """
        Clamp stroke parameters to fit in the range.
        :param val: float, used to clamp h/w and avoid big strokes
        :return:
        """
        self.x.data[:, :, :2] = torch.clamp(self.x.data[:, :, :2], 0.1, 1 - 0.1)  # position
        self.x.data[:, :, 2:4] = torch.clamp(self.x.data[:, :, 2:4], 0.1, val)  # width height
        self.x.data[:, :, 4] = torch.clamp(self.x.data[:, :, 4], 0.1, 1 - 0.1)  # theta
        self.x.data[:, :, 5:] = torch.clamp(self.x.data[:, :, 5:], 0, 1)  # color

    def __call__(self, paths: List):
        """
        Decompose the list of images in brushstroke parameters
        :param paths: list of image paths
        :return:
        """

        self.img = self.load_img(paths)

        PARAMS = torch.zeros([self.batch_size, 0, self.d], dtype=torch.float32, device=self.device)
        if self.args.canvas_color == "white":
            CANVAS_tmp = torch.ones([self.batch_size, 3, self.output_size, self.output_size]).to(
                self.device
            )
        else:
            CANVAS_tmp = torch.zeros([self.batch_size, 3, self.output_size, self.output_size]).to(
                self.device
            )

        for self.m_grid in range(1, self.max_divide + 1):
            self.img_batch = img2patches(img=self.img, m_grid=self.m_grid, s=self.output_size).to(
                self.device
            )
            CANVAS_tmp = img2patches(img=CANVAS_tmp, m_grid=self.m_grid, s=self.output_size)
            self.G_final_pred_canvas = CANVAS_tmp.clone()

            # ==== Initialize Parameters of Strokes
            self.initialize_params()
            self.x.requires_grad = True
            self.optimizer_x = optim.RMSprop([self.x], lr=self.lr, centered=True)

            self.step_id = 0
            for self.anchor_id in range(0, self.m_strokes_per_block):
                self.stroke_sampler(self.anchor_id)
                iters_per_stroke = int(self.args.max_iters / self.m_strokes_per_block)
                for i in range(iters_per_stroke):
                    self.G_pred_canvas = CANVAS_tmp
                    self.clamp(self.args.clip)

                    # update x
                    self.optimizer_x.zero_grad()
                    self._forward_pass()
                    self.compute_accuracy(verbose=self.args.verbose)
                    self._backward_x()

                    self.clamp(self.args.clip)

                    self.optimizer_x.step()
                    self.step_id += 1

            # === Stitch together the canvas, add parameters to stack
            CANVAS_tmp = patches2img(self.G_final_pred_canvas.detach(), m_grid=self.m_grid)
            v = self._normalize_strokes_and_reshape(self.x)
            PARAMS = torch.cat([PARAMS, v], axis=1)

        return PARAMS


if __name__ == "__main__":
    import argparse
    import numpy as np
    from utils import save_torch_img

    # settings
    parser = argparse.ArgumentParser(description="STYLIZED NEURAL PAINTING")
    parser.add_argument(
        "--canvas_color",
        type=str,
        default="black",
        metavar="str",
        help="canvas_color: [black, white] (default black)",
    )
    parser.add_argument(
        "--max_m_strokes",
        type=int,
        default=500,
        metavar="str",
        help="max number of strokes (default 500)",
    )
    parser.add_argument(
        "--max_divide",
        type=int,
        default=5,
        metavar="N",
        help="divide an image up-to max_divide x max_divide patches (default 5)",
    )
    parser.add_argument(
        "--beta_L1", type=float, default=1.0, help="weight for L1 loss (default: 1.0)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="learning rate for stroke searching (default: 0.005)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"/home/eperuzzo/snp-dataset/",
        metavar="str",
        help="dir to save painting results (default: ./output)",
    )
    # Added arguments
    parser.add_argument("--max_iters", default=500, type=int, help="Number of iteration per grid")
    parser.add_argument("--clip", default=0.9, type=float, help="clip h/w to this max size")
    parser.add_argument(
        "--verbose",
        action="store_false",
        default=True,
        help="Print loss during decomposition",
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Set number of images to be processed in parallel",
    )
    args = parser.parse_args()

    # Seed
    torch.manual_seed(0)
    np.random.seed(0)

    # === Main
    name = f"iters_{args.max_iters}_gird_{args.max_divide}_N_{args.max_m_strokes}_clip_{args.clip}"
    args.output_dir = os.path.join(args.output_dir, name)
    pt = ProgressiveOilPainter(args)

    frame_paths = glob.glob(os.path.join(args.input_path, "*.jpg"))

    bs = args.batch_size
    N = len(frame_paths) // bs

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    for j in range(N):
        curr_paths = frame_paths[j * bs : (j + 1) * bs]
        strokes = pt(curr_paths)

        # save strokes
        strokes = strokes.detach()
        for i in range(strokes.shape[0]):
            name = os.path.basename(curr_paths[i]).split(".")[0]
            pt.save_stroke_params(
                v=strokes[i], output_path=os.path.join(args.output_dir, f"{name}.pkl")
            )
        print(f" - Batch {j+1}/{N}: Successful!")
