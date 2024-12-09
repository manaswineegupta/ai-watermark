import argparse
import sys
import torch
from torch import nn
from PIL import Image
import numpy as np
import pickle
import os

from watermarkers.networks import FixedWatermarkers
from watermarkers.utils.styleTorch.TFtoTorchTools import (
    populate_module_params,
    load_tf_network,
)
from watermarkers.utils.styleTorch.networks import Generator


def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "model" or module == "layers":
            module = "watermarkers.networks.yu2.progan"
        return super().find_class(module, name)


class _LegacyPickle:
    Unpickler = _LegacyUnpickler


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)


class Yu2(FixedWatermarkers):
    def __init__(
        self,
        model_path,
        watermark_path,
        generator="StyleGAN2",
        dataset="CelebA",
        image_size=128,
        batch_size=64,
        device="cuda",
    ):
        assert generator in ["StyleGAN2", "PROGAN"]
        if generator == "StyleGAN2":
            assert dataset == "CelebA"
        assert dataset in ["CelebA", "LSUN"]

        gen_ext = ".pkl" if generator == "StyleGAN2" else ".pth"
        generator_path = os.path.join(model_path, generator + "_" + dataset + gen_ext)
        decoder_path = os.path.join(model_path, dataset + "_decoder.pth")
        watermark_path = os.path.join(watermark_path, dataset + "_watermark.npy")

        encoder_checkpoint = (
            load_tf_network(generator_path, "Encoder")
            if generator == "StyleGAN2"
            else generator_path
        )
        decoder_checkpoint = torch.load(decoder_path, map_location=device)
        watermark_length = decoder_checkpoint["dense.2.weight"].shape[0]
        self.sigmoid = nn.Sigmoid()

        super().__init__(
            encoder_checkpoint,
            decoder_checkpoint,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )
        self.gen_type = generator

    def encoder_inputs(self, batch_size):
        if self.gen_type == "PROGAN":
            return (
                hypersphere(torch.randn(batch_size, self.image_size, 1, 1)).to(
                    self.device
                ),
                {},
            )
        seeds = np.random.randint(10000, size=(batch_size,))
        z_i = torch.concat(
            [
                torch.from_numpy(
                    np.random.RandomState(seed).randn(1, self.encoder.z_dim)
                ).to(self.device)
                for seed in seeds
            ]
        )
        kwargs = {
            "c": torch.zeros([z_i.shape[0], self.encoder.c_dim], device=self.device),
        }
        return z_i, kwargs

    def post_process_raw(self, x):
        return None

    def process_encoder_input(self, x):
        return x

    def init_decoder(self, decoder_checkpoint):
        RevealNet = StegaStampDecoder(self.image_size, 3, self.watermark_length)
        RevealNet.load_state_dict(decoder_checkpoint)
        RevealNet = RevealNet.eval().requires_grad_(False)
        return RevealNet.requires_grad_(False).to(self.device).eval()

    def init_encoder(self, encoder_checkpoint):
        if isinstance(encoder_checkpoint, tuple):
            (kwargs, patterns) = encoder_checkpoint
            encoder = Generator(**kwargs).eval().requires_grad_(False)
            populate_module_params(encoder, *patterns)
        else:
            encoder = torch.load(
                encoder_checkpoint, pickle_module=_LegacyPickle, map_location="cpu"
            )
        return encoder.requires_grad_(False).to(self.device).eval()

    def _decode_batch_raw(self, x):
        return self.sigmoid(self.decoder(x))

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch).round()

    def _encode_batch(self, x_batch, msg_batch):
        z_i, kwargs = self.encoder_inputs(len(x_batch))
        images = self.encoder(z_i, **kwargs)
        return (images * 127.5 + 128).clamp(0, 255).div(255)
