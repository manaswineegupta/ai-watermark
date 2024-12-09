import torch
from torchvision import transforms
from compressai.zoo import bmshj2018_hyperprior, bmshj2018_factorized, cheng2020_anchor

from modules.attack import BaseAttack


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class VAE(BaseAttack):
    def __init__(
        self,
        evaluator,
        vae_type,
        input_dir,
        output_dir,
        quality=1,
        batch_size=1,
        device="cuda",
    ):
        super().__init__(
            evaluator=evaluator,
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            device=device,
        )
        assert vae_type in [
            "bmshj2018-factorized",
            "bmshj2018-hyperprior",
            "cheng2020-anchor",
        ]
        vae_type = vae_type.replace("-", "_")
        self.image_size = self.evaluator.image_size
        self.effective_size = next_power_of_2(self.image_size)

        self.optimizer = (
            eval(vae_type)(quality=quality, pretrained=True).eval().to(device)
        )

        self.upscaler = transforms.Resize(
            (self.effective_size, self.effective_size), antialias=None
        )
        self.downscaler = transforms.Resize(
            (self.image_size, self.image_size), antialias=None
        )

    def do_batch(self, x):
        x = self.upscaler(x)
        removed = torch.clamp(self.optimizer(x)["x_hat"], 0, 1).view(
            -1, 3, self.effective_size, self.effective_size
        )
        removed = self.downscaler(removed)
        return None, removed
