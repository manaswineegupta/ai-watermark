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
from .diffusion import InversableStableDiffusion, GuidedDiffusion
from watermarkers.networks import BaseWatermarker


class TREERING(BaseWatermarker):
    def __init__(
        self,
        checkpoint="stabilityai/stable-diffusion-2-1-base",
        diffuser="Stable",
        data_path="Gustavosta/Stable-Diffusion-Prompts",
        watermark_path=None,
        batch_size=64,
        device="cuda",
    ):
        assert diffuser in ["Stable", "imagenet"]
        checkpoint = os.path.join(checkpoint, diffuser)
        checkpoint = checkpoint + ".pth" if diffuser == "imagenet" else checkpoint
        image_size = 512 if diffuser != "imagenet" else 256
        pipe_constructor = (
            InversableStableDiffusion
            if "imagenet" not in checkpoint
            else GuidedDiffusion
        )
        with open(os.devnull, "w") as f:
            sys.stderr = f
            pipe = pipe_constructor(
                checkpoint,
                device,
            ).to(device)
            sys.stderr = sys.__stderr__

        pipe.unet.requires_grad_(False)

        pipe.set_progress_bar_config(leave=False)
        pipe.set_progress_bar_config(disable=True)
        watermark_path = (
            os.path.join(watermark_path, "imagenet_watermark.obj")
            if "imagenet" in checkpoint
            else os.path.join(watermark_path, "stable_watermark.obj")
        )
        super().__init__(
            pipe,
            pipe,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=64,
            batch_size=batch_size,
            device=device,
        )
        self.transforms = pipe.get_decoder_transforms()

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]
        self.encoder_kwargs = {
            "num_images_per_prompt": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "height": image_size,
            "width": image_size,
            "output_type": "numpy",
            "return_dict": False,
        }
        self.decoder_kwargs = {
            "guidance_scale": 1,
            "num_inference_steps": 50,
        }

        self.acceptance_thresh = 71

    def get_watermarking_mask(self, shape):
        watermarking_mask = torch.zeros(shape, dtype=torch.bool).to(self.device)
        np_mask = self.circle_mask(shape[-1], r=10)
        torch_mask = torch.tensor(np_mask).to(self.device)
        watermarking_mask[:, self.encoder.channel_idx()] = torch_mask
        return watermarking_mask

    def get_watermarking_pattern(self):
        gt_init = self.encoder.get_random_latents()
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(10, 0, -1):
            tmp_mask = self.circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(self.device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        return gt_patch

    @staticmethod
    def circle_mask(size=64, r=10):
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        return ((x - x0) ** 2 + (y[::-1] - y0) ** 2) <= r**2

    def init_encoder(self, encoder_checkpoint):
        return encoder_checkpoint

    def init_decoder(self, decoder_checkpoint):
        return decoder_checkpoint

    def get_secrets(self, input_dir, num_images, image_size=256):
        return super().get_raw_images(input_dir, num_images, image_size)

    def get_raw_images(self, input_dir, num_images, image_size=256):
        prompts = random.sample(list(range(len(self.dataset))), num_images)
        prompts = [self.dataset[i]["Prompt"] for i in prompts]
        init_latents_w = torch.concat(
            [self.encoder.get_random_latents() for i in range(num_images)]
        )
        return (init_latents_w, prompts, image_size)

    def post_process_raw(self, x):
        init_latents_w, prompts, image_size = x
        images = self.encoder(prompts, latents=init_latents_w, **self.encoder_kwargs)[
            0
        ].to(self.device)
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
        init_latents_w_fft = torch.fft.fftshift(
            torch.fft.fft2(init_latents_w), dim=(-1, -2)
        )
        mask = (
            self.watermarking_mask
            if len(self.watermarking_mask.shape) == len(init_latents_w_fft.shape)
            else self.watermarking_mask.unsqueeze(0)
        )
        mask = mask.repeat(
            init_latents_w_fft.shape[0], *([1] * (len(init_latents_w_fft.shape) - 1))
        )
        assert mask.shape == init_latents_w_fft.shape
        init_latents_w_fft[mask] = (
            self.watermark.clone().repeat(init_latents_w.shape[0], 1).view(-1)
        )
        init_latents_w = torch.fft.ifft2(
            torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))
        ).real
        return self.post_process_raw((init_latents_w, prompts, self.image_size))

    def init_watermark(self, watermark_path):
        if os.path.exists(watermark_path):
            with open(watermark_path, "rb") as f:
                watermarking_mask, pattern = pickle.load(f)
                self.watermarking_mask, pattern = torch.from_numpy(
                    watermarking_mask
                ).to(self.device), torch.from_numpy(pattern).to(self.device)
        else:
            self.watermarking_mask = self.get_watermarking_mask(
                self.encoder.get_random_latents().shape
            )
            pattern = self.get_watermarking_pattern()
            with open(watermark_path, "wb") as f:
                pickle.dump(
                    (
                        self.watermarking_mask.detach().cpu().numpy(),
                        pattern.detach().cpu().numpy(),
                    ),
                    f,
                )
        watermark = pattern[self.watermarking_mask].view(1, -1)
        self.watermark_length = watermark.shape[-1]
        return watermark

    def _decode_batch_raw(self, x):
        embeddings = torch.concat(
            [self.decoder.get_text_embedding("") for i in range(len(x))]
        )
        decoded = self.decoder.get_image_latents(
            self.transforms(x).to(embeddings.dtype).to(self.device), sample=False
        )
        decoded = self.decoder.forward_diffusion(
            latents=decoded,
            text_embeddings=embeddings,
            **self.decoder_kwargs,
        )
        decoded = torch.fft.fftshift(torch.fft.fft2(decoded), dim=(-1, -2))
        mask = (
            self.watermarking_mask
            if len(self.watermarking_mask.shape) == len(decoded.shape)
            else self.watermarking_mask.unsqueeze(0)
        )
        mask = mask.repeat(x.shape[0], *([1] * (len(decoded.shape) - 1)))
        return decoded[mask].view(x.shape[0], -1)

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def is_detected(self, accs):
        return accs < self.acceptance_thresh

    def stats(self, imgs, decoded, msg_batch):
        return self.err(decoded, msg_batch)
