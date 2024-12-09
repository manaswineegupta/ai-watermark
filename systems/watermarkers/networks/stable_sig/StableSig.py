import os
import random
from contextlib import nullcontext
import pickle
import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion

from watermarkers.networks import FixedWatermarkers


class StableSignature(FixedWatermarkers):
    def __init__(
        self,
        checkpoint="pretrained_models/watermarkers/stablesig/models",
        data_path="datasets/treering_prompts.obj",
        watermark_path="pretrained_models/watermarkers/stablesig/watermarks/watermark.npy",
        diffusion_type="base",  # base, v
        precision="autocast",
        watermark_length=48,
        batch_size=64,
        device="cuda",
    ):
        assert diffusion_type in ["base", "v"]
        model_path = (
            "v2-1_512-ema-pruned.ckpt"
            if diffusion_type == "base"
            else "v2-1_768-ema-pruned.ckpt"
        )
        conf_path = (
            "v2-inference.yaml" if diffusion_type == "base" else "v2-inference-v.yaml"
        )
        model_path = os.path.join(checkpoint, model_path)
        conf_path = os.path.join(checkpoint, conf_path)
        watermarker_path = os.path.join(checkpoint, "sd2_decoder.pth")
        extractor_path = os.path.join(checkpoint, "dec_48b_whit.torchscript.pt")
        image_size = 512 if diffusion_type == "base" else 768

        config = OmegaConf.load(conf_path).model
        config["params"]["unet_config"]["params"]["use_fp16"] = (
            True if precision == "autocast" else False
        )
        config["params"]["cond_stage_config"]["params"]["version"] = os.path.join(
            checkpoint, "open_clip", "open_clip_pytorch_model.bin"
        )

        super().__init__(
            (model_path, config, watermarker_path),
            extractor_path,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )
        self.precision_scope = (
            torch.autocast if precision == "autocast" else nullcontext
        )
        self.transforms = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.sampler = DDIMSampler(self.encoder, device=device)
        self.sampler_kwargs = {
            "S": 50,
            "shape": [4, image_size // 8, image_size // 8],
            "verbose": False,
            "unconditional_guidance_scale": 9.0,
            "eta": 0.0,
            "x_T": None,
        }

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]

    def __init_model(self, model_path, config):
        model_checkpoint = torch.load(model_path, map_location="cpu")["state_dict"]
        model = LatentDiffusion(**config.get("params", dict()))
        model.load_state_dict(model_checkpoint, strict=False)
        model.to(self.device)
        if self.device == "cpu":
            model.cond_stage_model.device = "cpu"
        model.eval()
        return model

    def init_encoder(self, encoder_checkpoint):
        model_path, config, watermarker_path = encoder_checkpoint
        encoder = self.__init_model(model_path, config)
        no_encoder = self.__init_model(model_path, config)
        watermarker_checkpoint = torch.load(watermarker_path)
        encoder.first_stage_model.load_state_dict(watermarker_checkpoint, strict=False)
        self.no_encoder = no_encoder.requires_grad_(False)
        return encoder.requires_grad_(False)

    def init_decoder(self, decoder_checkpoint):
        return torch.jit.load(decoder_checkpoint).to(self.device).eval()

    def get_secrets(self, input_dir, num_images, image_size=256):
        return super().get_raw_images(input_dir, num_images, image_size)

    def get_raw_images(self, input_dir, num_images, image_size=256):
        prompts = random.sample(list(range(len(self.dataset))), num_images)
        prompts = [self.dataset[i]["Prompt"] for i in prompts]
        n_batch = int(np.ceil(len(prompts) / self.batch_size))

        samples = []
        with torch.no_grad(), self.precision_scope(
            self.device
        ), self.encoder.ema_scope():
            for step in range(n_batch):
                prompts_i = prompts[
                    step * self.batch_size : (step + 1) * self.batch_size
                ]
                x_samples = self.sampler.sample(
                    conditioning=self.encoder.get_learned_conditioning(prompts_i),
                    unconditional_conditioning=self.encoder.get_learned_conditioning(
                        len(prompts_i) * [""]
                    ),
                    batch_size=len(prompts_i),
                    **self.sampler_kwargs,
                )[0]
                samples.extend([sample.unsqueeze(0) for sample in x_samples])
            samples = torch.concat(samples).view(len(prompts), *samples[0].shape[1:])
        return (samples, image_size)

    def post_process_raw(self, x):
        samples, image_size = x
        with torch.no_grad(), self.precision_scope(
            self.device
        ), self.no_encoder.ema_scope():
            return transforms.Resize((image_size, image_size), antialias=None)(
                torch.clamp(
                    (self.no_encoder.decode_first_stage(samples) + 1.0) / 2.0,
                    min=0.0,
                    max=1.0,
                )
            )

    def encode(self, x, with_grad=False):
        samples, orig_size = x
        encoded = []
        n_batch = int(np.ceil(len(samples) / self.batch_size))

        for step in range(n_batch):
            imgs = samples[step * self.batch_size : (step + 1) * self.batch_size]
            msg_batch = (
                self.watermark.repeat(len(imgs), 1)
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
            self.device
        )

    def get_watermarked_images(self, input_dir, num_images, image_size=256):
        raw_images = self.get_raw_images(input_dir, num_images, image_size=image_size)
        return (
            self.post_process_raw(raw_images).float(),
            self.encode(raw_images).float(),
        )

    def _encode_batch(self, x_batch, msg_batch):
        with self.precision_scope(self.device), self.encoder.ema_scope():
            return torch.clamp(
                (self.encoder.decode_first_stage(x_batch) + 1.0) / 2.0, min=0.0, max=1.0
            )

    def _decode_batch_raw(self, x):
        with self.precision_scope(self.device), self.encoder.ema_scope():
            return self.decoder(self.transforms(x))

    def _decode_batch(self, x_batch, msg_batch):
        return (self._decode_batch_raw(x_batch) > 0).float()
