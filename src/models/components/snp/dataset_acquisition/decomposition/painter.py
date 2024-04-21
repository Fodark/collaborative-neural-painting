import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio

from . import morphology, loss, utils, renderer
from .networks import *

import torch
import torch.optim as optim
from torch.nn.functional import interpolate, mse_loss
from torchvision.transforms.functional import crop


def _normalize(x, width):
    x = x * (width - 1) + 0.5
    return x.type(torch.LongTensor)


def get_bbox(st_point, window_size, max_h_w):
    """
    Args:
        st_point:
        window_size:

    Returns:
        coordinates to crop the input image
    """
    h_ws = (0.5 * window_size).type(torch.LongTensor)
    xc, yc = torch.split(st_point, 1, dim=-1)
    xc = xc.squeeze()
    yc = yc.squeeze()

    x1 = torch.where(xc - h_ws > 0, xc - h_ws, torch.zeros_like(xc))
    x2 = torch.where(xc + h_ws < max_h_w, xc + h_ws, torch.full_like(xc, torch.tensor(max_h_w)))

    # Fix
    x1[torch.where(x2 == max_h_w)] = max_h_w - 2 * h_ws[torch.where(x2 == max_h_w)]
    x2[torch.where(x1 == 0)] = 2 * h_ws[torch.where(x1 == 0)]

    # Y
    y1 = torch.where(yc - h_ws > 0, yc - h_ws, torch.zeros_like(yc))
    y2 = torch.where(yc + h_ws < max_h_w, yc + h_ws, torch.full_like(yc, torch.tensor(max_h_w)))
    # Fix
    y1[torch.where(y2 == max_h_w)] = max_h_w - 2 * h_ws[torch.where(y2 == max_h_w)]
    y2[torch.where(y1 == 0)] = 2 * h_ws[torch.where(y1 == 0)]

    return x1, x2, y1, y2


class PainterBase:
    def __init__(self, args):
        self.args = args
        self.rderr = renderer.Renderer(
            brush_paths=args.brush_paths,
            renderer=args.renderer,
            CANVAS_WIDTH=args.canvas_size,
            canvas_color=args.canvas_color,
        )

        # define G
        if args.gpu_id < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{args.gpu_id}")
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G, device=self.device).to(self.device)

        # define some other vars to record the training states
        self.x_ctt = None
        self.x_color = None
        self.x_alpha = None

        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.G_final_pred_canvas = torch.zeros(
            [1, 3, self.net_G.out_size, self.net_G.out_size]
        ).to(self.device)

        self.G_loss = torch.tensor(0.0)
        self.step_id = 0
        self.anchor_id = 0
        self.renderer_checkpoint_dir = args.renderer_checkpoint_dir
        self.output_dir = args.output_dir
        self.lr = args.lr

        # define the loss functions
        self._pxl_loss = loss.PixelLoss(p=1)
        self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=False)

        # some other vars to be initialized in child classes
        self.input_aspect_ratio = None
        self.img_path = None
        self.img_batch = None
        self.img_ = None
        self.final_rendered_images = None
        self.m_grid = None
        self.m_strokes_per_block = None

        if os.path.exists(self.args.dataset_feat_mean) and os.path.exists(
            self.args.dataset_feat_var
        ):
            self.mu = torch.load(self.args.dataset_feat_mean).unsqueeze(0)
            var = torch.load(self.args.dataset_feat_var)
            self.var_1 = (1 / var).unsqueeze(0)
            self.sigma_1 = torch.linalg.inv(torch.diag(var)).unsqueeze(0)

    def _load_checkpoint(self):

        # load renderer G
        if os.path.exists((os.path.join(self.renderer_checkpoint_dir, "last_ckpt.pt"))):
            print("loading renderer from pre-trained checkpoint...")
            # load the entire checkpoint
            checkpoint = torch.load(
                os.path.join(self.renderer_checkpoint_dir, "last_ckpt.pt"),
                map_location=self.device,
            )
            # update net_G states
            self.net_G.load_state_dict(checkpoint["model_G_state_dict"])
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print("pre-trained renderer does not exist...")
            exit()

    def _compute_acc(self):

        target = self.img_batch.detach()
        canvas = self.G_pred_canvas.detach()
        psnr = utils.cpt_batch_psnr(canvas, target, PIXEL_MAX=1.0)

        return psnr

    def _save_stroke_params(self, v):

        d_shape = self.rderr.d_shape
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha

        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape : d_shape + d_color]
        x_alpha = v[:, :, d_shape + d_color : d_shape + d_color + d_alpha]
        print("saving stroke parameters...")
        np.savez(
            os.path.join(self.output_dir, "strokes_params.npz"),
            x_ctt=x_ctt,
            x_color=x_color,
            x_alpha=x_alpha,
        )

    def _shuffle_strokes_and_reshape(self, v):

        grid_idx = list(range(self.m_grid**2))
        random.shuffle(grid_idx)
        v = v[grid_idx, :, :]
        v = np.reshape(np.transpose(v, [1, 0, 2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)

        return v

    def _render(self, v, save_jpgs=True, save_video=True):

        v = v[0, :, :]
        if self.args.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                out_h = int(self.args.canvas_size * self.input_aspect_ratio)
                out_w = self.args.canvas_size
            else:
                out_h = self.args.canvas_size
                out_w = int(self.args.canvas_size / self.input_aspect_ratio)
        else:
            out_h = self.args.canvas_size
            out_w = self.args.canvas_size

        file_name = os.path.join(self.output_dir, self.img_path.split("/")[-1][:-4])

        if save_video:
            video_writer = cv2.VideoWriter(
                file_name + "_animated.mp4",
                cv2.VideoWriter_fourcc(*"MP4V"),
                40,
                (out_w, out_h),
            )

        # print('rendering canvas...')
        self.rderr.create_empty_canvas()
        for i in range(v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                self.rderr.draw_stroke()
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(
                    file_name + "_rendered_stroke_" + str((i + 1)).zfill(4) + ".png",
                    this_frame,
                )
            if save_video:
                video_writer.write((this_frame[:, :, ::-1] * 255.0).astype(np.uint8))

        if save_jpgs:
            print("saving input photo...")
            out_img = cv2.resize(self.img_, (out_w, out_h), cv2.INTER_AREA)
            plt.imsave(file_name + "_input.png", out_img)

        final_rendered_image = np.copy(this_frame)
        if save_jpgs:
            print("saving final rendered result...")
            plt.imsave(file_name + "_final.png", final_rendered_image)

        return final_rendered_image

    def _normalize_strokes(self, v):

        v = np.array(v.detach().cpu())

        if self.rderr.renderer in ["watercolor", "markerpen"]:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, ...
            xs = np.array([0, 4])
            ys = np.array([1, 5])
            rs = np.array([6, 7])
        elif self.rderr.renderer in ["oilpaintbrush", "rectangle"]:
            # xc, yc, w, h, theta ...
            xs = np.array([0])
            ys = np.array([1])
            rs = np.array([2, 3])
        else:
            raise NotImplementedError("renderer [%s] is not implemented" % self.rderr.renderer)

        for y_id in range(self.m_grid):
            for x_id in range(self.m_grid):
                y_bias = y_id / self.m_grid
                x_bias = x_id / self.m_grid
                v[y_id * self.m_grid + x_id, :, ys] = (
                    y_bias + v[y_id * self.m_grid + x_id, :, ys] / self.m_grid
                )
                v[y_id * self.m_grid + x_id, :, xs] = (
                    x_bias + v[y_id * self.m_grid + x_id, :, xs] / self.m_grid
                )
                v[y_id * self.m_grid + x_id, :, rs] /= self.m_grid

        return v

    def initialize_params(self):

        self.x_ctt = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block, self.rderr.d_shape
        ).astype(np.float32)
        self.x_ctt = torch.tensor(self.x_ctt).to(self.device)

        self.x_color = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block, self.rderr.d_color
        ).astype(np.float32)
        self.x_color = torch.tensor(self.x_color).to(self.device)

        self.x_alpha = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block, self.rderr.d_alpha
        ).astype(np.float32)
        self.x_alpha = torch.tensor(self.x_alpha).to(self.device)

    def stroke_sampler(self, anchor_id):

        if anchor_id == self.m_strokes_per_block:
            return

        err_maps = torch.sum(
            torch.abs(self.img_batch - self.G_final_pred_canvas), dim=1, keepdim=True
        ).detach()

        for i in range(self.m_grid * self.m_grid):
            this_err_map = err_maps[i, 0, :, :].cpu().numpy()
            ks = int(this_err_map.shape[0] / 8)
            this_err_map = cv2.blur(this_err_map, (ks, ks))
            this_err_map = this_err_map**4
            this_img = self.img_batch[i, :, :, :].detach().permute([1, 2, 0]).cpu().numpy()

            self.rderr.random_stroke_params_sampler(err_map=this_err_map, img=this_img)

            self.x_ctt.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[0 : self.rderr.d_shape]
            )
            self.x_color.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[
                    self.rderr.d_shape : self.rderr.d_shape + self.rderr.d_color
                ]
            )
            self.x_alpha.data[i, anchor_id, :] = torch.tensor(self.rderr.stroke_params[-1])

    def _compute_kl(self, params):
        pass

    def _compute_log_prob(self, params):
        pass

    def _backward_x(self):

        self.G_loss = 0
        pxl_loss = self._pxl_loss(canvas=self.G_final_pred_canvas, gt=self.img_batch)
        self.G_loss += self.args.beta_L1 * pxl_loss
        if self.args.with_ot_loss:
            self.G_loss += self.args.beta_ot * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.img_batch
            )
        if self.args.with_kl_loss:
            kl_loss = self._compute_log_prob(params=self.x)
            self.G_loss += self.args.beta_kl * kl_loss
        self.G_loss.backward()

    def _forward_pass(self):

        self.x = torch.cat([self.x_ctt, self.x_color, self.x_alpha], dim=-1)

        v = torch.reshape(
            self.x[:, 0 : self.anchor_id + 1, :],
            [self.m_grid * self.m_grid * (self.anchor_id + 1), -1, 1, 1],
        )
        self.G_pred_foregrounds, self.G_pred_alphas = self.net_G(v)

        self.G_pred_foregrounds = morphology.Dilation2d(m=1)(self.G_pred_foregrounds)
        self.G_pred_alphas = morphology.Erosion2d(m=1)(self.G_pred_alphas)

        self.G_pred_foregrounds = torch.reshape(
            self.G_pred_foregrounds,
            [
                self.m_grid * self.m_grid,
                self.anchor_id + 1,
                3,
                self.net_G.out_size,
                self.net_G.out_size,
            ],
        )
        self.G_pred_alphas = torch.reshape(
            self.G_pred_alphas,
            [
                self.m_grid * self.m_grid,
                self.anchor_id + 1,
                3,
                self.net_G.out_size,
                self.net_G.out_size,
            ],
        )

        for i in range(self.anchor_id + 1):
            G_pred_foreground = self.G_pred_foregrounds[:, i]
            G_pred_alpha = self.G_pred_alphas[:, i]
            self.G_pred_canvas = G_pred_foreground * G_pred_alpha + self.G_pred_canvas * (
                1 - G_pred_alpha
            )

        self.G_final_pred_canvas = self.G_pred_canvas


########################################################################################################################
# Modify this class
class Painter(PainterBase):
    def __init__(self, args):
        super(Painter, self).__init__(args=args)
        self.args = args

        self._load_checkpoint()
        self.net_G.eval()
        print(f"Painter created, weights form: {args.renderer_checkpoint_dir}, eval mode: True")

    def manual_set_number_strokes_per_block(self, id):
        self.m_strokes_per_block = self.manual_strokes_per_block[id]

    def stroke_parser(self):

        total_blocks = 0
        for i in range(0, self.max_divide + 1):
            total_blocks += i**2

        return int(self.max_m_strokes / total_blocks)

    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print_fn = False
        if print_fn:
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
        vis2 = utils.patches2img(self.G_final_pred_canvas, self.m_grid).clip(min=0, max=1)
        if self.args.disable_preview:
            pass
        else:
            cv2.namedWindow("G_pred", cv2.WINDOW_NORMAL)
            cv2.namedWindow("input", cv2.WINDOW_NORMAL)
            cv2.imshow("G_pred", vis2[:, :, ::-1])
            cv2.imshow("input", self.img_[:, :, ::-1])
            cv2.waitKey(1)

    def _render(
        self,
        v,
        path=None,
        canvas_start=None,
        save_jpgs=False,
        save_video=False,
        save_gif=False,
        highlight_border=False,
        color_border=(1, 0, 0),
    ):

        self.rderr.highlight_border = highlight_border
        self.rderr.color_border = color_border
        v = v[
            0, :, : self.rderr.d
        ]  # if we add additional information, make sure to use only needed parms
        if self.args.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                out_h = int(self.args.canvas_size * self.input_aspect_ratio)
                out_w = self.args.canvas_size
            else:
                out_h = self.args.canvas_size
                out_w = int(self.args.canvas_size / self.input_aspect_ratio)
        else:
            out_h = self.args.canvas_size
            out_w = self.args.canvas_size

        if save_video:
            """
            video_writer = cv2.VideoWriter(
                path + '_animated.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10,
                (out_w, out_h))
            """
            video_writer = imageio.get_writer(path + "_animated.mp4", codec="libx264", fps=3)

        # print('rendering canvas...')
        if canvas_start is None:
            self.rderr.create_empty_canvas()
        else:
            self.rderr.canvas = canvas_start

        alphas = []
        for i in range(v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                alpha = self.rderr.draw_stroke()
                alphas.append(alpha)
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(os.path.join(path, str(i) + ".jpg"), this_frame)
            if save_video:
                # video_writer.write((this_frame[:,:,::-1] * 255.).astype(np.uint8))
                video_writer.append_data(np.uint8(this_frame * 255))
            if save_gif:
                if i == 0:
                    gif_imgs = []
                    gif_imgs.append(np.uint8(this_frame * 255.0))
                else:
                    gif_imgs.append(np.uint8(this_frame * 255.0))

        final_rendered_image = np.copy(this_frame)
        # if save_jpgs:
        #     print('saving final rendered result...')
        #     plt.imsave(path + '_final.png', final_rendered_image)

        if save_gif:
            print("saving gif ...")
            imageio.mimsave(path + ".gif", gif_imgs, duration=0.1)
        if save_video:
            video_writer.close()
        return final_rendered_image, np.concatenate(alphas)

    def get_checked_strokes(self, v):
        v = v[0, :, :]
        checked_strokes = []
        for i in range(v.shape[0]):
            if self.check_stroke(v[i, :]):
                checked_strokes.append(v[i, :][None, :])
        return np.concatenate(checked_strokes, axis=0)[
            None, :, :
        ]  # restore the 1, n, parmas dimension for consistency

    @staticmethod
    def check_stroke(inp):
        """
        Copy and pasetd form renderder.py
        They have a threshold on the min size of the brushstorkes
        """

        r_ = max(inp[2], inp[3])  # check width and height, as in the original code
        if r_ > 0.025:
            return True
        else:
            return False

    def _save_stroke_params(self, v, path):

        d_shape = self.rderr.d_shape
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha

        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape : d_shape + d_color]
        x_alpha = v[:, :, d_shape + d_color : d_shape + d_color + d_alpha]
        x_layer = v[:, :, d_shape + d_color + d_alpha :]

        path = os.path.join(path, "strokes_params.npz")
        print(f"saving stroke parameters at {path}...")
        np.savez(path, x_ctt=x_ctt, x_color=x_color, x_alpha=x_alpha, x_layer=x_layer)

    def _shuffle_strokes_and_reshape(self, v):

        grid_idx = list(range(self.m_grid**2))
        random.shuffle(grid_idx)
        v = v[grid_idx, :, :]
        v = np.reshape(np.transpose(v, [1, 0, 2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)

        return v, np.array(grid_idx)

    def clamp(self, val):
        # Modification, use a different clamp for width and height
        pos = torch.clamp(self.x_ctt.data[:, :, :2], 0.1, 1 - 0.1)
        theta = torch.clamp(self.x_ctt.data[:, :, 4], 0.1, 1 - 0.1)
        size = torch.empty_like(self.x_ctt.data[:, :, 2:4])
        for i in range(pos.shape[0]):
            size[i] = torch.clamp(self.x_ctt.data[i, :, 2:4], min=0.1, max=val)

        # Put all back together
        self.x_ctt.data = torch.cat([pos, size, theta.unsqueeze(-1)], dim=-1)
        self.x_color.data = torch.clamp(self.x_color.data, 0, 1)
        self.x_alpha.data = torch.clamp(self.x_alpha.data, 0, 1)

    def train(self):
        # -------------------------------------------------------------------------------------------------------------
        # Set parameters
        # self.max_divide = args.max_divide
        # self.max_m_strokes = args.max_m_strokes

        # manually set the number of strokes, use more strokes at the beginning
        self.manual_strokes_per_block = self.args.manual_storkes_params
        self.m_strokes_per_block = None  # self.stroke_parser()

        self.max_divide = max(self.manual_strokes_per_block.keys())
        self.m_grid = 1

        self.img_path = self.args.img_path
        self.img_ = cv2.imread(self.args.img_path, cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.img_ = cv2.resize(
            self.img_,
            (
                self.net_G.out_size * self.max_divide,
                self.net_G.out_size * self.max_divide,
            ),
            cv2.INTER_AREA,
        )
        # --------------------------------------------------------------------------------------------------------------
        print("begin drawing...")

        clamp_schedule = self.args.clamp_schedule
        PARAMS = np.zeros([1, 0, self.rderr.d + 2], np.float32)  # +2 to save layer information

        if self.rderr.canvas_color == "white":
            CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(self.device)
        else:
            CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(self.device)

        for self.m_grid in self.manual_strokes_per_block.keys():

            self.img_batch = utils.img2patches(self.img_, self.m_grid, self.net_G.out_size).to(
                self.device
            )
            self.G_final_pred_canvas = CANVAS_tmp

            self.manual_set_number_strokes_per_block(self.m_grid)
            self.initialize_params()
            self.x_ctt.requires_grad = True
            self.x_color.requires_grad = True
            self.x_alpha.requires_grad = True
            utils.set_requires_grad(self.net_G, False)

            self.optimizer_x = optim.RMSprop(
                [self.x_ctt, self.x_color, self.x_alpha], lr=self.lr, centered=True
            )

            self.step_id = 0
            for self.anchor_id in range(0, self.m_strokes_per_block):
                self.stroke_sampler(self.anchor_id)
                iters_per_stroke = int(500 / self.m_strokes_per_block)
                for i in range(iters_per_stroke):
                    self.G_pred_canvas = CANVAS_tmp

                    # update x
                    self.optimizer_x.zero_grad()
                    self.clamp(val=clamp_schedule[self.m_grid])

                    self._forward_pass()
                    self._drawing_step_states()
                    self._backward_x()

                    self.clamp(val=clamp_schedule[self.m_grid])

                    self.optimizer_x.step()
                    self.step_id += 1

            v = self._normalize_strokes(self.x)
            v, idx_grid = self._shuffle_strokes_and_reshape(v)

            # Add layer information
            layer_info = np.full((1, v.shape[1], 1), self.m_grid)
            grid_info = np.repeat(idx_grid, self.m_strokes_per_block)[
                None, :, None
            ]  # repeat for each storke, add dim 0 and -1
            v = np.concatenate([v, layer_info, grid_info], axis=-1)

            # Add on previous parmas
            PARAMS = np.concatenate([PARAMS, v], axis=1)
            CANVAS_tmp, _ = self._render(PARAMS, save_jpgs=False, save_video=False)
            CANVAS_tmp = utils.img2patches(CANVAS_tmp, self.m_grid + 1, self.net_G.out_size).to(
                self.device
            )

        PARAMS = self.get_checked_strokes(PARAMS)
        # final_rendered_image, alphas = self._render(PARAMS, save_jpgs=False, save_video=False)
        return PARAMS

    def inference(
        self,
        strokes,
        output_path=None,
        order=None,
        canvas_start=None,
        save_jpgs=False,
        save_video=False,
        save_gif=False,
        hilight=False,
    ):

        if order is not None:
            strokes = strokes[:, order, :]

        if output_path is None:
            img, alphas = self._render(strokes, canvas_start=canvas_start)
            return img, alphas
        else:
            _ = self._render(
                strokes,
                path=output_path,
                canvas_start=canvas_start,
                save_jpgs=save_jpgs,
                save_video=save_video,
                save_gif=save_gif,
                highlight_border=hilight,
            )

    ###############################################################################################################
    # ADDED FOR INTERACTIVE TASK
    def initialize_params_predict(self, bs):

        self.x_ctt = np.random.rand(bs, self.m_strokes_per_block, self.rderr.d_shape).astype(
            np.float32
        )
        self.x_ctt = torch.tensor(self.x_ctt).to(self.device)

        self.x_color = np.random.rand(bs, self.m_strokes_per_block, self.rderr.d_color).astype(
            np.float32
        )
        self.x_color = torch.tensor(self.x_color).to(self.device)

        self.x_alpha = np.random.rand(bs, self.m_strokes_per_block, self.rderr.d_alpha).astype(
            np.float32
        )
        self.x_alpha = torch.tensor(self.x_alpha).to(self.device)

    def _compute_log_prob(self, params):
        bs = params.shape[0]
        params = params[:, :, :11]
        color = 0.5 * (params[:, :, 5:8] + params[:, :, 8:11])
        params = torch.cat((params[:, :, :5], color), dim=-1)
        x = torch.cat((self.ctx, params), dim=1)
        x = utils.compute_features(x)

        mu = self.mu.repeat(bs, 1).to(x.device)
        weights = self.var_1.repeat(bs, 1).to(x.device)
        weights[:, 2 * 250 : 5 * 250] = 0
        log_prob = mse_loss(x, mu, reduction="none") * weights
        log_prob = torch.mean(torch.sum(log_prob, dim=-1))
        """
        sigma_1 = self.sigma_1[:, :].repeat(bs, 1, 1).to(x.device)
        mu = self.mu.to(x.device)
        diff = (x-mu).unsqueeze(1)
        log_prob = diff.bmm(sigma_1).bmm(diff.transpose(1,2))
        log_prob = torch.mean(log_prob)
        """
        return log_prob

    def _normalize_strokes_predict(self, v):
        v = v.detach().cpu()

        v[:, :, 0] = (v[:, :, 0] * self.ws[:, None] + self.x1[:, None]) / self.args.canvas_size
        v[:, :, 1] = (v[:, :, 1] * self.ws[:, None] + self.y1[:, None]) / self.args.canvas_size
        v[:, :, 2] = (v[:, :, 2] * self.ws[:, None]) / self.args.canvas_size
        v[:, :, 3] = (v[:, :, 3] * self.ws[:, None]) / self.args.canvas_size

        v = v[:, :, :11]  # remove alpha
        final_color = 0.5 * (v[:, :, 5:8] + v[:, :, 8:])
        final_params = torch.cat((v[:, :, :5], final_color), dim=-1)
        return final_params

    def predict(self, img, CANVAS_tmp=None):

        bs = img.size(0)
        self.max_divide = 1
        self.m_grid = np.uint8(np.sqrt(bs))
        self.m_strokes_per_block = 8
        self.max_strokes = 8

        iters_per_stroke = (
            self.args.n_iters_per_strokes
        )  # number of optimizations steps for a single stroke

        clamp_value = torch.empty(self.ws.shape, dtype=torch.float32)
        clamp_value[torch.where(self.ws == 128)] = 0.4
        clamp_value[torch.where(self.ws == 64)] = 0.3
        clamp_value[torch.where(self.ws == 32)] = 0.25
        # --------------------------------------------------------------------------------------------------------------
        # print('begin drawing...')

        self.img_batch = (
            img  # utils.img2patches(img, self.m_grid, self.net_G.out_size).to(self.device)
        )
        self.G_final_pred_canvas = CANVAS_tmp

        self.initialize_params_predict(bs=bs)
        self.x_ctt.requires_grad = True
        self.x_color.requires_grad = True
        self.x_alpha.requires_grad = True
        utils.set_requires_grad(self.net_G, False)

        self.optimizer_x = optim.RMSprop(
            [self.x_ctt, self.x_color, self.x_alpha], lr=self.lr, centered=True
        )

        self.step_id = 0
        for self.anchor_id in range(0, self.max_strokes):
            self.stroke_sampler(self.anchor_id)
            for i in range(iters_per_stroke):
                # print(f'Anchor: {self.anchor_id}, iter: {i} / {iters_per_stroke}')
                self.G_pred_canvas = CANVAS_tmp
                # update x
                self.optimizer_x.zero_grad()
                self.clamp(clamp_value)
                self._forward_pass()
                self._drawing_step_states()
                self._backward_x()
                # self.clamp(clamp_value)
                self.optimizer_x.step()
                self.step_id += 1

        return self._normalize_strokes_predict(self.x)

    def get_ctx(self, ctx):
        # Location of the last stroke
        """
        x_start, y_start = ctx[:, -1, :2]
        x_start = _normalize(x_start, self.args.canvas_size)
        y_start = _normalize(y_start, self.args.canvas_size)
        """
        start = _normalize(ctx[:, -1, :2], self.args.canvas_size)

        # Select window size based on average stroke area
        area = torch.mean(ctx[:, :, 2] * ctx[:, :, 3], dim=-1)

        windows_size = torch.empty_like(area)

        windows_size[torch.where(area < 0.004)] = 32
        windows_size[torch.where(torch.logical_and(area >= 0.004, area < 0.01))] = 64
        windows_size[torch.where(area >= 0.01)] = 128
        return start, windows_size.type(torch.LongTensor)

    def generate(self, data):
        original = data["img"]
        canvas_start = data["canvas"]
        strokes_ctx = data["strokes_ctx"]

        bs = original.shape[0]
        out = []

        if self.args.with_kl_loss:
            self.ctx = data["strokes_ctx"]

        st_point, self.ws = self.get_ctx(strokes_ctx)
        self.x1, self.x2, self.y1, self.y2 = get_bbox(st_point, self.ws, self.args.canvas_size)

        # crop
        curr_imgs = []
        curr_canvas = []
        for b in range(bs):
            curr_imgs.append(
                interpolate(
                    crop(
                        original[b][None],
                        top=self.y1[b],
                        left=self.x1[b],
                        height=self.ws[b],
                        width=self.ws[b],
                    ),
                    size=(self.net_G.out_size, self.net_G.out_size),
                )
            )
            curr_canvas.append(
                interpolate(
                    crop(
                        canvas_start[b][None],
                        top=self.y1[b],
                        left=self.x1[b],
                        height=self.ws[b],
                        width=self.ws[b],
                    ),
                    size=(self.net_G.out_size, self.net_G.out_size),
                )
            )

        curr_imgs = torch.cat(curr_imgs, dim=0)
        curr_canvas = torch.cat(curr_canvas, dim=0)

        # Predict strokes
        snp_preds = self.predict(curr_imgs, curr_canvas)
        return snp_preds.numpy().astype("float32")
