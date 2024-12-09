import argparse
import sys
import torch
from torch import nn
from PIL import Image
import numpy as np

from watermarkers.networks import RandomWatermarkers
from watermarkers.utils.styleTorch.TFtoTorchTools import (
    populate_module_params,
    load_tf_network,
)
from watermarkers.utils.styleTorch.networks_watermark import Generator, Decoder


class Yu1(RandomWatermarkers):
    def __init__(
        self,
        network_path="StyleGAN2_150k_CelebA_128x128.pkl",
        watermark_path="STyleGAN_CELEBA_watermark.npy",
        watermark_length=128,
        image_size=128,
        batch_size=64,
        device="cuda",
    ):
        encoder_checkpoint = load_tf_network(network_path, "WatermarkEnc")
        decoder_checkpoint = load_tf_network(network_path, "Decoder")
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

    def post_process_raw(self, x):
        return None

    def process_encoder_input(self, x):
        return x

    def init_decoder(self, decoder_checkpoint):
        (kwargs, patterns) = decoder_checkpoint
        decoder = Decoder(**kwargs).eval().requires_grad_(False)
        populate_module_params(decoder, *patterns)
        return decoder.requires_grad_(False).to(self.device).eval()

    def init_encoder(self, encoder_checkpoint):
        (kwargs, patterns) = encoder_checkpoint
        encoder = Generator(**kwargs).eval().requires_grad_(False)
        populate_module_params(encoder, *patterns)
        return encoder.requires_grad_(False).to(self.device).eval()

    def _decode_batch_raw(self, x):
        return self.sigmoid(self.decoder(x, None)[1])

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch).round()

    def _encode_batch(self, x_batch, msg_batch):
        seeds = np.random.randint(10000, size=(len(x_batch)))
        z_i = torch.concat(
            [
                torch.from_numpy(
                    np.random.RandomState(seed).randn(1, self.encoder.z_dim)
                ).to(self.device)
                for seed in seeds
            ]
        )
        label = torch.zeros([z_i.shape[0], self.encoder.c_dim], device=self.device)
        images = self.encoder(
            z_i,
            msg_batch,
            label,  # truncation_psi=0.5, noise_mode="random"
        )
        return (images * 127.5 + 128).clamp(0, 255).div(255)
