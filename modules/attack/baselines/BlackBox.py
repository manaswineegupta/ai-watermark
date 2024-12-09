import os
from PIL import Image
from io import BytesIO
import torch
from torch import nn
import kornia
from torchvision import transforms
from diffusers import LDMSuperResolutionPipeline

from modules.attack import BaseAttack


class BlackBoxAttack(nn.Module):
    def __init__(self, transLst):
        super().__init__()
        self.transforms = transLst

    def forward(self, x):
        return self.transforms(x)


class CropLayer(BlackBoxAttack):
    def __init__(self, image_size, ratio=0.9):
        assert ratio >= 0.9 and ratio <= 1
        cropped_sz = int(ratio * image_size)
        transLst = transforms.Compose(
            [
                transforms.CenterCrop((cropped_sz, cropped_sz)),
                transforms.Resize((image_size, image_size), antialias=None),
            ]
        )
        super().__init__(transLst)


class JPEGLayer(BlackBoxAttack):
    def __init__(self, image_size, quality):
        # assert quality >= 80 and quality <= 200 and isinstance(quality, int)

        def compress(x):
            outputIoStream = BytesIO()
            transforms.ToPILImage()(x.squeeze().cpu().detach()).save(
                outputIoStream, "JPEG", quality=quality, optimice=True
            )
            outputIoStream.seek(0)
            return (
                transforms.ToTensor()(Image.open(outputIoStream).convert("RGB"))
                .to(x.device)
                .view(-1, 3, x.shape[-1], x.shape[-1])
            )

        def JPEGcompression(x):
            return torch.concat([compress(x_i) for x_i in x]).view(
                -1, 3, x.shape[-1], x.shape[-1]
            )

        transLst = transforms.Compose(
            [
                transforms.Lambda(JPEGcompression),
            ]
        )

        super().__init__(transLst)

    def forward(self, x):
        return super().forward(x).view(*x.shape).to(x.device)


class SuperResolutionLayer(BlackBoxAttack):
    def __init__(self, image_size, ro):
        assert ro >= 0.125 and ro <= 0.5

        pipline = LDMSuperResolutionPipeline.from_pretrained(
            "pretrained_models/super_resolution",
            # "CompVis/ldm-super-resolution-4x-openimages"
        )

        def SR(x):
            x_hat = x
            while x_hat.shape[-1] < image_size:
                x_hat = (
                    torch.from_numpy(
                        pipline(2 * x_hat - 1, output_type="numpy", return_dict=False)[
                            0
                        ]
                    )
                    .to(x.device)
                    .permute(0, 3, 1, 2)
                )
            return x_hat

        transLst = transforms.Compose(
            [
                transforms.Resize(
                    (int(image_size * ro // 1), int(image_size * ro // 1)),
                    antialias=None,
                ),
                transforms.Lambda(SR),
                transforms.Resize((image_size, image_size), antialias=None),
            ]
        )
        super().__init__(transLst)


class IDLayer(BlackBoxAttack):
    def __init__(self, image_size):
        super().__init__(transforms.Compose([transforms.Lambda(lambda x: x)]))


class BlurLayer(BlackBoxAttack):
    def __init__(self, image_size, kernel_size, sigma=(0.1, 2.0)):
        trans_blur = transforms.GaussianBlur(kernel_size, sigma=sigma)

        def blur(x):
            out = trans_blur(x)
            return torch.clamp(out, 0, 1)

        super().__init__(transforms.Compose([transforms.Lambda(blur)]))


class GuidedBlurLayer(BlackBoxAttack):
    def __init__(self, image_size, kernel_size, sigma=(0.1, 2.0), color_sigma=0.1):
        def blur(x):
            out = kornia.filters.bilateral_blur(
                x, (kernel_size, kernel_size), color_sigma, sigma
            )
            return torch.clamp(out, 0, 1)

        super().__init__(transforms.Compose([transforms.Lambda(blur)]))


class NoiseLayer(BlackBoxAttack):
    def __init__(self, image_size, sigma):
        assert sigma >= 0 and sigma <= 0.05

        def noise(x):
            out = x + torch.normal(mean=0, std=sigma, size=x.shape).to(x.device)
            return torch.clamp(out, 0, 1)

        super().__init__(transforms.Compose([transforms.Lambda(noise)]))


class QuantizeLayer(BlackBoxAttack):
    def __init__(self, image_size, strength):
        # assert strength >= 0.5 and strength <= 1

        def quantize(x):
            x = x * 255
            out = strength * torch.floor(x / strength)
            out = out / 255
            return torch.clamp(out, 0, 1)

        transLst = transforms.Compose([transforms.Lambda(quantize)])
        super().__init__(transLst)


class BaseLineAttack(BaseAttack):
    def __init__(
        self,
        evaluator,
        input_dir,
        output_dir,
        batch_size=1,
        device="cuda",
        **layer_args,
    ):
        super().__init__(
            evaluator=evaluator,
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            device=device,
        )
        self.layer = self.Layer()(image_size=self.image_size, **layer_args).to(
            self.device
        )

    def do_batch(self, x):
        return None, self.layer(x)


class GuidedBlur(BaseLineAttack):
    def Layer(self):
        return GuidedBlurLayer


class Quantize(BaseLineAttack):
    def Layer(self):
        return QuantizeLayer


class Blur(BaseLineAttack):
    def Layer(self):
        return BlurLayer


class Noise(BaseLineAttack):
    def Layer(self):
        return NoiseLayer


class SuperResolution(BaseLineAttack):
    def Layer(self):
        return SuperResolutionLayer


class ID(BaseLineAttack):
    def Layer(self):
        return IDLayer

    def post_process_removed(self, imgs):
        return None


class JPEG(BaseLineAttack):
    def Layer(self):
        return JPEGLayer


class Crop(BaseLineAttack):
    def Layer(self):
        return CropLayer
