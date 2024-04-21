import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage

from .utils import read_img, Dilation2d, Erosion2d


@torch.jit.script
def draw_canvas(
        canvas: torch.Tensor,
        brushes: torch.Tensor,
        colors: torch.Tensor,
        alphas: torch.Tensor,
):
    for i in range(brushes.shape[1]):
        colored_brush = brushes[:, i] * colors[:, i]
        canvas = colored_brush * alphas[:, i] + canvas * (1 - alphas[:, i])
    return canvas

@torch.jit.script
def draw_canvas_progressive(
        canvas: torch.Tensor,
        brushes: torch.Tensor,
        colors: torch.Tensor,
        alphas: torch.Tensor,
):
    images = []
    for i in range(brushes.shape[1]):
        colored_brush = brushes[:, i] * colors[:, i]
        canvas = colored_brush * alphas[:, i] + canvas * (1 - alphas[:, i])
        images.append(canvas[0])
    return images


class Renderer(nn.Module):
    def __init__(self, canvas_size, half_precision=False, morphology=True):
        super().__init__()
        self.H = canvas_size[0]
        self.W = canvas_size[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.half_precision = half_precision
        brush_paths = {
            "large_vertical": "src/models/components/snp/brushes/brush_fromweb2_large_vertical.png",
            "large_horizontal": "src/models/components/snp/brushes/brush_fromweb2_large_horizontal.png",
        }
        self.register_buffer("meta_brushes", self.load_meta_brushes(brush_paths))
        self.register_buffer("meta_alphas", self.load_meta_alphas(brush_paths))

        self.meta_brushes = self.meta_brushes.to(self.device)
        self.meta_alphas = self.meta_alphas.to(self.device)

        self.morphology = morphology
        if self.morphology:
            self.dilation = Dilation2d(m=1).to(self.device)
            self.erosion = Erosion2d(m=1).to(self.device)
        # self.dilation = Dilation2d(1, 1).to(self.device)
        # self.erosion = Erosion2d(1, 1).to(self.device)

    def load_meta_brushes(self, paths):
        brush_large_vertical = read_img(paths["large_vertical"], "L", h=self.H, w=self.W)
        brush_large_horizontal = read_img(paths["large_horizontal"], "L", h=self.H, w=self.W)
        brushes = torch.cat([brush_large_vertical, brush_large_horizontal], dim=0)
        # Stack the brushes and convert to float16
        if self.half_precision:
            brushes = brushes.half()
        else:
            brushes = brushes.float()
        return brushes

    def load_meta_alphas(self, paths):
        brush = self.load_meta_brushes(paths)
        alphas = brush > 0
        if self.half_precision:
            alphas = alphas.half()
        else:
            alphas = alphas.float()
        return alphas

    def render_single_strokes(self, strokes: torch.Tensor):
        """
        Render each stroke without composing the result on the canvas
        :param strokes: [bs x 8] parameters of the input strokes
        :return:
        """
        if len(strokes.shape) == 3:
            strokes = strokes.flatten(0, 1)
            print("- Strokes reshaped before rendering")
        bs, dim = strokes.shape
        colors, foregrounds, alphas = self.strokes2brushes(strokes)
        foregrounds = self.dilation(foregrounds)
        alphas = self.erosion(alphas)
        foregrounds = foregrounds * colors
        foregrounds = foregrounds.reshape(bs, 3, self.H, self.W)
        alphas = alphas.reshape(bs, 1, self.H, self.W)

        return foregrounds, alphas

    def draw_on_canvas(self, strokes: torch.Tensor, canvas_color="black", progressive=False):
        """
        Render each stroke and compose the result on the output canvas
        :param strokes: [bs x N x 8] parameters of the input strokes
        :param canvas_color: either black or white
        :return: [bs x 3 x canvas_size x canvas_size] rendered result
        """
        bs, L, num = strokes.shape
        # Approximate the strokes to occupy less memory
        strokes = strokes.reshape(bs * L, num)
        # Create the brushes on the canvas
        colors, foregrounds, alphas = self.strokes2brushes(strokes)
        if self.morphology:
            foregrounds = self.dilation(foregrounds)
            alphas = self.erosion(alphas)
        # Reshape
        colors = torch.reshape(colors, (bs, L, 3, 1, 1))
        foregrounds = torch.reshape(foregrounds, (bs, L, 1, self.H, self.W))
        alphas = torch.reshape(alphas, (bs, L, 1, self.H, self.W))

        # ==== Define the starting canvas ===
        if canvas_color == "black":
            canvas = torch.zeros(bs, 3, self.H, self.W, device=foregrounds.device)
        else:
            canvas = torch.ones(bs, 3, self.H, self.W, device=foregrounds.device)
        # ==== Draw the strokes ====
        if progressive:
            images = draw_canvas_progressive(canvas, foregrounds, colors, alphas)
            return images
        else:
            canvas = draw_canvas(canvas, foregrounds, colors, alphas)
            return canvas

    def strokes2brushes(self, strokes: torch.Tensor):
        """
        Render the strokes, warping scaling and placing the meta-brushes as described by the input strokes parameters.
        :param strokes:
        :return:
        """
        # strokes: [b, 8]
        N = strokes.shape[0]
        x0, y0, w, h, theta = strokes[:, :5].T
        colors = strokes[:, 5:8, None, None]
        # Meta brushes: [Large Vertical, Large Horizontal]
        brushes_idx = (h <= w).long()
        brushes = self.meta_brushes[brushes_idx].to(strokes.device)
        alphas = self.meta_alphas[brushes_idx].to(strokes.device)
        # ==== Affine transformation ====
        rad_theta = theta * torch.pi
        sin_theta = torch.sin(rad_theta)
        cos_theta = torch.cos(rad_theta)

        warp_00 = cos_theta / w
        warp_01 = sin_theta * self.H / (self.W * w)
        warp_02 = (1 - 2 * x0) * warp_00 + (1 - 2 * y0) * warp_01
        warp_10 = -sin_theta * self.W / (self.H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * warp_11 - (1 - 2 * x0) * -warp_10
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        # Stack the affine transformation matrix and convert into float16
        warp = torch.stack([warp_0, warp_1], dim=1)
        # Convert the tensors to the correct datatype
        warp = warp.type(brushes.dtype)
        # Apply the affine transformation
        grid = torch.nn.functional.affine_grid(warp, (N, 1, self.H, self.W), align_corners=False)
        brushes = torch.nn.functional.grid_sample(brushes, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)

        return colors, brushes, alphas


if __name__ == "__main__":
    from torchvision.utils import save_image

    # root_dir = Path("/data/ndallasen/inp-dataset")
    renderer = Renderer((256, 256), half_precision=True)
    # strokes = torch.rand((32, 180, 8), device="cuda" if torch.cuda.is_available() else "cpu")
    strokes = torch.load("strokes.pt", map_location="cpu").to("cuda" if torch.cuda.is_available() else "cpu").unsqueeze(
        0).half()
    strokes.clamp_(0, 1)

    print(strokes.shape, "min", strokes.min(), "max", strokes.max())

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()
    # for _ in trange(1000):
    canvas = renderer.draw_on_canvas(strokes + 1e-3, canvas_color="black")
    save_image(canvas, "test.png")
    # save the canvas
    # end.record()
    # torch.cuda.synchronize()

    # print(f"Rendering time: {start.elapsed_time(end) / 1000:.3f}s")
    # print(f"Rendering time: {time.time() - start_time:.3f}s")
    # for macrocategory in root_dir.iterdir():
    #     for category in macrocategory.iterdir():
    #         # pick a random files containing .reordered in the name
    #         for file in category.iterdir():
    #             if "reordered" in file.name:
    #                 print(file)
    #                 strokes_torch = pkl.load(open(file, "rb"))
    #                 strokes_torch = torch.from_numpy(strokes_torch).float()
    #                 images = []
    #                 for i in trange(1, len(strokes_torch) + 1):
    #                     image = renderer.draw_on_canvas(strokes_torch[:i].unsqueeze(0))
    #                     image = image.squeeze(0).numpy() * 255
    #                     image = image.transpose(1, 2, 0).astype(np.uint8)
    #                     images.append(image)
    #                 imageio.mimsave(f"output/{file.name}.gif", images, duration=0.05)
    #                 break
