import os
import sys
import copy
import random
import numpy as np
import pickle
import torch
from torchvision import transforms
import logging

logging.basicConfig(level=logging.ERROR)
from watermarkers.networks import BaseWatermarker
from .utils_prc import KeyGen, Encode, Detect, recover_posteriors
from .inversion import stable_diffusion_pipe, generate, exact_inversion


class Prc(BaseWatermarker):
    def __init__(
        self,
        checkpoint="stabilityai/stable-diffusion-2-1-base",
        data_path="Gustavosta/Stable-Diffusion-Prompts",
        keys_path="encode_decode_keys.pkl",
        watermark_path="prc_codeword.pkl",
        watermark_size=4*64*64,
        batch_size=64,
        device="cuda",
    ):
        image_size = 512 
        hf_cache_dir = "cache/" ## change this maybe
        pipe = stable_diffusion_pipe(solver_order=1, model_id=checkpoint, cache_dir=hf_cache_dir)
        pipe.set_progress_bar_config(disable=True)

        super().__init__(
            pipe,
            pipe,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=4*64*64, 
            batch_size=batch_size,
            device=device,
        )

        self.encoding_key, self.decoding_key = self.init_prc_keys(keys_path)
        self.watermark_size = watermark_size

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]


    def init_encoder(self, encoder_checkpoint):
        return encoder_checkpoint

    def init_decoder(self, decoder_checkpoint):
        return decoder_checkpoint

    def init_prc_keys(self, path):
        with open(path, 'rb') as f:
            keys = pickle.load(f)
        return keys

    def get_secrets(self, input_dir, num_images, image_size=256):
        return super().get_raw_images(input_dir, num_images, image_size)

    def get_raw_images(self, input_dir, num_images, image_size=256): 
        prompts = random.sample(list(range(len(self.dataset))), num_images)
        prompts = [self.dataset[i]["Prompt"] for i in prompts]
        init_latents_batch = []
        for i in range(num_images):
            latent = np.random.randn(self.watermark_size).reshape(1, 4, 64, 64)
            init_latents_batch.append(latent)
        init_latents_w = torch.tensor(init_latents_batch, dtype=torch.float64)
        #init_latents_w = torch.randn((num_images, self.watermark_size), dtype=torch.float64)
        #init_latents_b = torch.concat(
            #[torch.from_numpy(np.random.randn(self.watermark_size)).to(dtype=torch.float64) for i in range(num_images)]#########
        #)
        return (init_latents_w, prompts, image_size)

    def post_process_raw(self, x): 
        init_latents_w, prompts, image_size = x
        images = []
        for i in range(len(prompts)):
            current_prompt = prompts[i]
            current_latent = init_latents_w[i] 
            orig_image, _, _ = generate(prompt=current_prompt,init_latents=current_latent,num_inference_steps=50,solver_order=1,pipe=self.encoder)
            orig_image = transforms.ToTensor()(orig_image).view(1, 3, image_size, image_size)
            images.append(orig_image)
        images = torch.cat(images, dim=0).view(-1, 3, image_size, image_size).to(self.device)
        return transforms.Resize((image_size, image_size), antialias=None)(images)

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
        new_latents = (self.watermark * torch.abs(init_latents_w.reshape(self.watermark.shape))).reshape(init_latents_w.shape)
        return self.post_process_raw((new_latents, prompts, self.image_size))

    def init_watermark(self, watermark_path): 
        with open(watermark_path, 'rb') as f:
            watermark = pickle.load(f)
        watermark = watermark.to(dtype=torch.float64)
        return watermark

    def _decode_batch_raw(self, x):
        reversed_prc_batch = []
        for img in x:
            img = transforms.ToPILImage()(img)
            reversed_latents = exact_inversion(img, prompt='', test_num_inference_steps=50, inv_order=0, pipe=self.decoder)
            reversed_prc = recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(1.5)).flatten().cpu()
            reversed_prc_batch.append(reversed_prc)
        reversed_prc_batch = torch.stack(reversed_prc_batch)
        return reversed_prc_batch

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def is_detected(self, acc):
        return acc >= 0

    def stats(self, imgs, decoded, msg_batch):
        stats_batch = []
        for reversed_prc in decoded:
            detect_stats = Detect(self.decoding_key, reversed_prc)
            stats_batch.append(detect_stats)
        return torch.tensor(stats_batch, dtype=torch.float64)

    