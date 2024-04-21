import math
import torch
import random

from typing import List, Tuple

from src import utils

log = utils.get_pylogger(__name__)


class HorizontalFlip(object):
    """Horizontally flip the given strokes randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, strokes):
        """
        Args:
            strokes (torch.Tensor): Strokes tensor to be flipped.

        Returns:
            torch.Tensor: Randomly flipped strokes.
        """
        augmented = 0
        if random.random() < self.p:
            augmented = 1
            strokes[:, 0] = 1 - strokes[:, 0]
        return strokes, [augmented]


class VerticalFlip(object):
    """Vertically flip the given strokes randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, strokes):
        """
        Args:
            strokes (torch.Tensor): Strokes tensor to be flipped.

        Returns:
            torch.Tensor: Randomly flipped strokes.
        """
        augmented = 0
        if random.random() < self.p:
            augmented = 1
            strokes[:, 1] = 1 - strokes[:, 1]
        return strokes, [augmented]


class Rotate(object):
    """Rotate the given strokes randomly with a given probability.

    Args:
        p (float): probability of the image being rotated. Default value is 0.5
    """

    def __init__(self, p=0.5, max_theta=2 * math.pi):
        self.p = p
        self.max_theta = max_theta

    def __call__(self, strokes):
        """
        Args:
            strokes (torch.Tensor): Strokes tensor to be rotated.

        Returns:
            torch.Tensor: Randomly rotated strokes.
        """
        theta = random.random() * self.max_theta
        theta_applied = 0

        if random.random() < self.p:
            theta_applied = theta / (2 * math.pi)
            # find the center of the strokes
            x_min = torch.min(strokes[:, 0])
            x_max = torch.max(strokes[:, 0])
            y_min = torch.min(strokes[:, 1])
            y_max = torch.max(strokes[:, 1])
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # translate the strokes to the origin
            strokes[:, 0] -= x_center
            strokes[:, 1] -= y_center
            rotation_matrix = torch.tensor(
                [
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)],
                ],
                dtype=torch.float,
                device=strokes.device,
            )
            strokes[:, :2] = torch.matmul(strokes[:, :2], rotation_matrix)
            strokes[:, 4] = (strokes[:, 4] + (theta / (2 * math.pi))) % 1

            # translate the strokes back to the center
            strokes[:, 0] += x_center
            strokes[:, 1] += y_center

        return strokes, [theta_applied]


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers.
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
            Should be non negative numbers.
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
            Should be non negative numbers.
        hue (float): How much to jitter hue. hue_factor
            is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        if brightness < 0:
            raise ValueError("brightness should be non negative number")
        if contrast < 0:
            raise ValueError("contrast should be non negative number")
        if saturation < 0:
            raise ValueError("saturation should be non negative number")

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, strokes):
        """
        Args:
            strokes (torch.Tensor): Strokes tensor to be color jittered.

        Returns:
            torch.Tensor: Randomly color jittered strokes.
        """

        alphas_applied = [0, 0, 0]
        # consider that color is defined on strokes[:, 5:]
        # brightness
        if self.brightness > 0:
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            alphas_applied[0] = alpha
            strokes[:, 5:] *= alpha

        # contrast
        if self.contrast > 0:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            alphas_applied[1] = alpha
            strokes[:, 5:] *= alpha
            strokes[:, 5:] += (1 - alpha) / 3

        # saturation
        if self.saturation > 0:
            alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
            alphas_applied[2] = alpha
            gray = strokes[:, 5:].mean(dim=1, keepdim=True)
            strokes[:, 5:] = alpha * (strokes[:, 5:] - gray) + gray

        # clamp color values
        strokes[:, 5:] = torch.clamp(strokes[:, 5:], 0, 1)
        return strokes, alphas_applied


class Translate(object):
    """Translate the given strokes randomly with a given probability.

    Args:
        p (float): probability of the image being translated. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, strokes):
        """
        Args:
            strokes (torch.Tensor): Strokes tensor to be translated.

        Returns:
            torch.Tensor: Randomly translated strokes.
        """
        shift_applied = [0, 0]
        if random.random() < self.p:
            # find the center of the strokes
            x_shift = random.uniform(-0.1, 0.1)
            y_shift = random.uniform(-0.1, 0.1)
            shift_applied = [x_shift, y_shift]
            strokes[:, 0] += x_shift
            strokes[:, 1] += y_shift

            # delete the strokes where the element 0 or 1 is out of the range [0, 1]
            strokes = strokes[
                torch.logical_and(
                    torch.logical_and(strokes[:, 0] >= 0, strokes[:, 0] <= 1),
                    torch.logical_and(strokes[:, 1] >= 0, strokes[:, 1] <= 1),
                )
            ]

        return strokes, shift_applied


class RandomCrop(object):
    """Crop the given strokes randomly with a given probability.

    Args:
        p (float): probability of the image being cropped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, strokes):
        """
        Args:
            strokes (torch.Tensor): Strokes tensor to be cropped.

        Returns:
            torch.Tensor: Randomly cropped strokes.
        """
        crop_applied = [0, 0, 0]
        if random.random() < self.p:
            if strokes.shape[0] == 0:
                return strokes, crop_applied
            # find the center of the strokes
            zoom_factor = random.uniform(1.1, 1.3)
            crop_applied[0] = 1 - zoom_factor

            # find center of the strokes
            x_min = torch.min(strokes[:, 0])
            x_max = torch.max(strokes[:, 0])
            y_min = torch.min(strokes[:, 1])
            y_max = torch.max(strokes[:, 1])
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # translate the strokes to the origin
            strokes[:, 0] -= x_center
            strokes[:, 1] -= y_center

            # construct the zoom matrix
            zoom_matrix = torch.tensor(
                [[zoom_factor, 0], [0, zoom_factor]],
                dtype=torch.float,
                device=strokes.device,
            )
            strokes[:, :2] = torch.matmul(strokes[:, :2], zoom_matrix)
            strokes[:, 2:4] *= zoom_factor

            # translate the strokes back to the center with some random shift
            random_x_shift = random.uniform(-0.1, 0.1)
            random_y_shift = random.uniform(-0.1, 0.1)
            crop_applied[1] = random_x_shift
            crop_applied[2] = random_y_shift
            strokes[:, 0] += x_center + random_x_shift
            strokes[:, 1] += y_center + random_y_shift

            # delete the strokes where the element 0 or 1 is out of the range [0, 1]
            strokes = strokes[
                torch.logical_and(
                    torch.logical_and(strokes[:, 0] >= 0, strokes[:, 0] <= 1),
                    torch.logical_and(strokes[:, 1] >= 0, strokes[:, 1] <= 1),
                )
            ]

        return strokes, crop_applied


def get_augmentations():
    augmentations: List = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(max_theta=math.pi / 4),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        Translate(),
        RandomCrop(),
    ]

    def augment(strokes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_strokes = strokes
        augmentation_total_params = []
        for augmentation in augmentations:
            strokes, augmentation_params = augmentation(strokes)
            augmentation_total_params.extend(augmentation_params)

        if strokes.shape[0] == 0:
            log.warning("Strokes are empty after augmentation")
            # return original strokes and a vector of zeros
            return original_strokes, torch.zeros(
                len(augmentation_total_params),
                dtype=original_strokes.dtype,
                device=strokes.device,
            )
        return strokes, torch.tensor(augmentation_total_params, device=strokes.device)

    return augment


if __name__ == "__main__":
    src_file = "/data/shared/ndallasen/datasets/inp/ducks-2levels/train_0_0.pt"
    strokes_original = torch.load(src_file)
    augs = get_augmentations()
    for _ in range(10):
        strokes_aug, params = augs(strokes_original.float())
        print(list(params.shape))
        print(params.tolist())
        print("------------------")
    # torch.save(strokes_aug, "train_0_0_aug.pt")

    # from src.models.components.snp.renderer import Renderer
    # renderer = Renderer((256, 256))
    # strokes = torch.stack([strokes_original, strokes_aug])
    # img = renderer.draw_on_canvas(strokes)
    # from torchvision.utils import save_image
    # save_image(img, "test.png")
