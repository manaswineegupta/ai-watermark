import os
import sys
import copy
import random
import numpy as np
import pickle
import torch
from torchvision import transforms
from torch.special import erf
from diffusers import DPMSolverMultistepScheduler
import logging

logging.basicConfig(level=logging.ERROR)
from watermarkers.networks import BaseWatermarker

from .inverse_stable_diffusion import InversableStableDiffusionPipeline


class Prc(BaseWatermarker):
    def __init__(
        self,
        checkpoint="stabilityai/stable-diffusion-2-1-base",
        data_path="Gustavosta/Stable-Diffusion-Prompts",
        keys_path="encode_decode_keys.pkl",
        watermark_path="prc_codeword.pkl",
        watermark_size=4 * 64 * 64,
        batch_size=64,
        device="cuda",
    ):
        image_size = 512
        hf_cache_dir = "cache/"
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=1,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            checkpoint,
            scheduler=scheduler,
            torch_dtype=torch.float32,
            cache_dir=hf_cache_dir,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        super().__init__(
            pipe,
            pipe,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=4 * 64 * 64,
            batch_size=batch_size,
            device=device,
        )

        self.encoding_key, self.decoding_key = self.init_prc_keys(keys_path)
        self.watermark_size = watermark_size

        self.tform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        )

        self.decoder_inv = False
        self.prompt = ""
        self.test_num_inference_steps = 50
        self.guidance_scale = 3.0
        self.inv_order = 0
        self.variances = 1.5

        _, parity_check_matrix, one_time_pad, self.fpr, self.noise_rate, _, _, _, t = (
            self.decoding_key
        )
        self.one_time_pad = torch.from_numpy(np.array(one_time_pad, dtype=float)).to(
            self.device
        )
        self.parity_indices = torch.from_numpy(
            parity_check_matrix.indices.reshape(parity_check_matrix.shape[0], t)
        ).to(self.device)

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]

    def init_encoder(self, encoder_checkpoint):
        return encoder_checkpoint

    def init_decoder(self, decoder_checkpoint):
        return decoder_checkpoint

    def init_prc_keys(self, path):
        with open(path, "rb") as f:
            keys = pickle.load(f)
        return keys

    def get_secrets(self, input_dir, num_images, image_size=256):
        return super().get_raw_images(input_dir, num_images, image_size)

    def get_raw_images(self, input_dir, num_images, image_size=256):
        prompts = random.sample(list(range(len(self.dataset))), num_images)
        prompts = [self.dataset[i]["Prompt"] for i in prompts]
        init_latents_w = torch.randn(
            (num_images, 4, 64, 64), dtype=torch.float64, device=self.device
        )
        return (init_latents_w, prompts, image_size)

    def post_process_raw(self, x):
        return self.encoder(
            x[1],
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.test_num_inference_steps,
            height=x[2],
            width=x[2],
            latents=x[0],
        ).view(-1, 3, x[2], x[2])

    def encode(self, x, with_grad=False):
        init_latents_w, prompts, orig_size = x
        encoded = []
        n_batch = int(np.ceil(len(init_latents_w) / self.batch_size))

        for step in range(n_batch):
            latents_i = init_latents_w[
                step * self.batch_size : (step + 1) * self.batch_size
            ].to(self.device)
            prompts_i = prompts[step * self.batch_size : (step + 1) * self.batch_size]
            imgs = (latents_i, prompts_i, orig_size)
            msg_batch = (
                self.watermark.repeat(latents_i.shape[0], 1)
                if self.watermark is not None
                else None
            )
            if not with_grad:
                with torch.no_grad():
                    encoded_image_batch = self._encode_batch(imgs, msg_batch)
            else:
                encoded_image_batch = self._encode_batch(imgs, msg_batch)
            encoded.append(encoded_image_batch)
        encoded = torch.concat(encoded).view(-1, 3, self.image_size, self.image_size)
        return transforms.Resize((orig_size, orig_size), antialias=None)(encoded).to(
            init_latents_w.device
        )

    def get_watermarked_images(self, input_dir, num_images, image_size=256):
        raw_images = self.get_raw_images(input_dir, num_images, image_size=image_size)
        return self.post_process_raw(raw_images), self.encode(raw_images)

    def _encode_batch(self, x_batch, msg_batch):
        init_latents_w, prompts, image_size = x_batch
        new_latents = (
            (self.watermark * torch.abs(init_latents_w.reshape(self.watermark.shape)))
            .reshape(init_latents_w.shape)
            .to(self.device)
        )
        return self.post_process_raw((new_latents, prompts, self.image_size))

    def init_watermark(self, watermark_path):
        with open(watermark_path, "rb") as f:
            watermark = pickle.load(f)
        watermark = watermark.to(dtype=torch.float64, device=self.device)
        return watermark

    def _decode_batch_raw(self, x):
        x = 2.0 * self.tform(x) - 1.0
        image_latents = (
            self.decoder.decoder_inv(x)
            if self.decoder_inv
            else self.decoder.get_image_latents(x, sample=False)
        )
        text_embeddings_tuple = self.decoder.encode_prompt(
            self.prompt, x.device, 1, self.guidance_scale > 1.0, None
        )
        text_embeddings = torch.cat(
            [text_embeddings_tuple[1], text_embeddings_tuple[0]]
        )
        reversed_latents = (
            self.decoder.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.test_num_inference_steps,
                inverse_opt=(self.inv_order != 0),
                inv_order=self.inv_order,
            )
            .to(torch.float64)
            .flatten(start_dim=1)
        )
        return erf(
            reversed_latents / np.sqrt(2 * self.variances * (1 + self.variances))
        )

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def stats(self, imgs, decoded, msg_batch):
        posteriors = (
            (1 - 2 * self.noise_rate)
            * (1 - 2 * self.one_time_pad.unsqueeze(0))
            * decoded
        )
        Pi = torch.prod(posteriors[:, self.parity_indices], dim=-1)
        log_plus = torch.log((1 + Pi) / 2)
        log_minus = torch.log((1 - Pi) / 2)
        log_prod = log_plus + log_minus
        const = 0.5 * torch.sum(
            log_plus.square() + log_minus.square() - 0.5 * log_prod.square(), dim=-1
        )
        threshold = torch.sqrt(2 * const * np.log(1 / self.fpr)) + 0.5 * log_prod.sum(
            dim=-1
        )
        return (log_plus.sum(dim=-1) - threshold).double()

    def threshold(self, n):
        return 0
