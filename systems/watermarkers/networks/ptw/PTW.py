import os
import numpy as np
import torch

from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.watermarking_key.wm_key import WatermarkingKey
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory
from src.arguments.model_args import ModelArgs
from src.models.model_factory import ModelFactory

from watermarkers.networks import BaseWatermarker


class PtW(BaseWatermarker):
    def __init__(
        self,
        generator_ckpt,
        watermark_key_ckpt,
        image_size=256,
        batch_size=16,
        device="cuda",
    ):
        ckpt_fn = os.path.abspath(watermark_key_ckpt)
        wm_key_args: WatermarkingKeyArgs = torch.load(ckpt_fn)[
            WatermarkingKeyArgs.WM_KEY_ARGS_KEY
        ]
        wm_key_args.key_ckpt = ckpt_fn
        env_args = EnvArgs(
            device=device, batch_size=batch_size, eval_batch_size=batch_size
        )

        decoder_checkpoint = (wm_key_args, env_args)
        watermark_length = wm_key_args.bitlen

        model_args = torch.load(generator_ckpt)[ModelArgs.MODEL_ARGS_KEY]
        model_args.model_ckpt = generator_ckpt
        encoder_checkpoint = model_args

        super().__init__(
            encoder_checkpoint,
            decoder_checkpoint,
            watermark_path=None,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )

    def post_process_raw(self, x):
        return None

    def init_encoder(self, encoder_checkpoint):
        return ModelFactory.from_model_args(encoder_checkpoint)[0]

    def _encode_batch(self, x_batch, msg_batch):
        x = self.encoder.generate(len(x_batch))[1]  # , truncation_psi=0.7)[1]
        return (x + 1) / 2

    def init_decoder(self, decoder_checkpoint):
        wm_key_args, env_args = decoder_checkpoint
        wm_key: WatermarkingKey = (
            WatermarkingKeyFactory.from_watermark_key_args(
                wm_key_args, env_args=env_args
            )
            .eval()
            .requires_grad_(False)
        )
        return wm_key

    def init_watermark(self, watermark_path):
        return (
            WatermarkingKey.str_to_bits(self.decoder.wm_key_args.message)
            .unsqueeze(0)[:, : self.watermark_length]
            .to(self.device)
        )

    def process_encoder_input(self, x):
        return x

    def _decode_batch_raw(self, x):
        return self.decoder.extract(x * 2 - 1, sigmoid=True)

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch).round()
