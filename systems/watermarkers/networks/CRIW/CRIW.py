import torch
from torchvision import transforms
from PIL import Image
from watermarkers.networks import RandomWatermarkers
from .encoder import Encoder
from .resnet18 import ResNet
from scipy.stats import binom
from scipy.stats import norm
import math
import numpy as np


class CRIW(RandomWatermarkers):
    def __init__(
        self,
        weights_path="adversarial.pth",
        watermark_path="criw_watermark.npy",
        watermark_length=30, ######
        image_size=128,
        batch_size=64,
        device="cuda",
    ):

        self.tform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.checkpoint = torch.load(weights_path)
        self.range = 0.4
        self.sigma = 0.1
        self.alpha = 0.001
        self.tau = 0.83

        super().__init__(
            '',
            '',
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )

    def init_encoder(self, encoder_checkpoint):
        encoder = Encoder().to(self.device)
        encoder.load_state_dict(self.checkpoint['enc-model'])
        encoder.eval()
        return encoder

    def init_decoder(self, decoder_checkpoint):
        decoder = ResNet().to(self.device)
        decoder.load_state_dict(self.checkpoint['dec-model'])
        decoder.eval()
        return decoder

    def _encode_batch(self, x_batch, msg_batch):
        images = self.tform(x_batch) 
        secret =  torch.cat([self.watermark] * x_batch.shape[0], dim=0)   
        output = self.encoder(images, secret)
        output = output * 0.5 + 0.5 
        return output.view(-1, 3, self.image_size, self.image_size)

    def _decode_batch_raw(self, x):
        images = self.tform(x) 
        decoded = self.decoder(images)
        return (decoded >= 0.5).float()

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def stats(self, imgs, decoded, msg_batch): ############ DOESN'T WORK
        ### CORRECLY IMPLEMENT THIS REGERESSION BASED SMOOTHNING
        ### FIX THIS !!!
        perturbation_array = np.linspace(0, self.range, int(self.range * 100))
        certified_ba_array = np.zeros((decoded.shape[0], len(perturbation_array)))
        bitacc = torch.mean((decoded == msg_batch).float(), dim=1)

        for b in range(decoded.shape[0]):
            for idx, epsilon in enumerate(perturbation_array):
                k = self.certify(epsilon) 

                if k == -1:
                    continue
                else:
                    certified_ba_array[b, idx] = bitacc[b]

        print(certified_ba_array)

        return certified_ba_array

    def certify(self, epsilon): ####### DOES NOT WORK
        ### FIX THIS!!!
        p_lower = norm.cdf(-1 * (epsilon / self.sigma))
        upper_bound_of_klower = math.floor(p_lower)

        if upper_bound_of_klower == 0:
            return -1

        for i in range(upper_bound_of_klower):
            c_ = 1 - binom.cdf(i, 1, p_lower)
            if c_ < 1 - self.alpha:
                break

        if i == upper_bound_of_klower:
            return -1
        return (i - 1)


  

