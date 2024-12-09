import os
import torch
from torch import nn
from torchvision import transforms
import numpy as np

from watermarkers.networks import RandomWatermarkers


class ConvBNRelu(nn.Module):
    # A block of Convolution, Batch Normalization, and ReLU activation
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    # Receives an image and decodes the watermark. The input image may be watermarked or non-watermarked. Moreover,
    # the input image may have various kinds of noise applied to it, such as JpegCompression, Gaussian blur, and so on.
    # See Noise layers for more.
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.watermark_length))
        layers.append(ConvBNRelu(self.channels, config.watermark_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.watermark_length, config.watermark_length)

    def forward(self, watermarked_image):
        decoded_watermark = self.layers(watermarked_image)
        # The output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make the
        # tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        decoded_watermark.squeeze_(3).squeeze_(2)
        decoded_watermark = self.linear(decoded_watermark)

        return decoded_watermark


class Encoder(nn.Module):
    ### Embed a watermark into the original image and output the watermarked image.
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(
            self.conv_channels + 3 + config.watermark_length, self.conv_channels
        )
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, original_image, watermark):
        # First, add two dummy dimensions in the end of the watermark.
        # This is required for the .expand to work correctly.
        expanded_watermark = watermark.unsqueeze(-1)
        expanded_watermark.unsqueeze_(-1)
        expanded_watermark = expanded_watermark.expand(-1, -1, self.H, self.W)
        encoded_image = self.conv_layers(original_image)

        # Concatenate expanded watermark and the original image.
        concat = torch.cat([expanded_watermark, encoded_image, original_image], dim=1)
        watermarked_image = self.after_concat_layer(concat)
        watermarked_image = self.final_layer(watermarked_image)

        return watermarked_image


class Configuration:
    def __init__(
        self,
        H: int,
        W: int,
        watermark_length: int,
        encoder_blocks: int,
        encoder_channels: int,
        decoder_blocks: int,
        decoder_channels: int,
        use_discriminator: bool,
        use_vgg: bool,
        discriminator_blocks: int,
        discriminator_channels: int,
        decoder_loss: float,
        encoder_loss: float,
        adversarial_loss: float,
        enable_fp16: bool = False,
    ):
        self.H = H
        self.W = W
        self.watermark_length = watermark_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16


class HiDDeN(RandomWatermarkers):
    def __init__(
        self,
        checkpoint,
        watermark_path,
        image_size=128,
        watermark_length=30,
        batch_size=32,
        device="cuda",
    ):
        checkpoint = torch.load(checkpoint, map_location=device)
        encoder_checkpoint = checkpoint["enc-model"]
        decoder_checkpoint = checkpoint["dec-model"]
        super().__init__(
            encoder_checkpoint,
            decoder_checkpoint,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )
        self.transforms = transforms.Compose(
            [
                transforms.CenterCrop((image_size, image_size)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def get_conf(self):
        return Configuration(
            H=self.image_size,
            W=self.image_size,
            watermark_length=self.watermark_length,
            encoder_blocks=4,
            encoder_channels=64,
            decoder_blocks=7,
            decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3,
            discriminator_channels=64,
            decoder_loss=1,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=False,
        )

    def process_encoder_input(self, x):
        return x

    def init_decoder(self, decoder_checkpoint):
        decoder = Decoder(self.get_conf())
        decoder.load_state_dict(decoder_checkpoint)
        return decoder.requires_grad_(False).to(self.device).eval()

    def init_encoder(self, encoder_checkpoint):
        encoder = Encoder(self.get_conf())
        encoder.load_state_dict(encoder_checkpoint)
        return encoder.requires_grad_(False).to(self.device).eval()

    def _decode_batch_raw(self, x):
        return self.decoder(x * 2 - 1)

    def _decode_batch(self, x_batch, msg_batch):
        return torch.clamp(torch.round(self._decode_batch_raw(x_batch)), 0, 1)

    def _encode_batch(self, x_batch, msg_batch):
        encoded_image_batch = self.encoder(self.transforms(x_batch), msg_batch)
        encoded_image_batch = (encoded_image_batch + 1) / 2
        return encoded_image_batch.mul(255).add_(0.5).clamp_(0, 255).div(255)
