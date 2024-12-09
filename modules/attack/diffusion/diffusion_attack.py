import numpy as np
import torch
import torch as th
from torchvision import transforms
from diffusers import UNet2DModel

from modules.attack import BaseAttack

from . import gaussian_diffusion as gd
from .gaussian_diffusion import GaussianDiffusion
from .unet import UNetModel


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(1000 if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


class BaseDiffusion(torch.nn.Module):
    def __init__(
        self,
        model,
        betas,
        sample_step=1,
        t=0.2,
        device="cuda",
    ):
        super().__init__()
        self.t = t
        self.sample_step = sample_step
        self.model = model
        self.betas = betas
        self.device = device

    def diffuse(self, x, t):
        pass

    def forward(self, img):
        img = (img - 0.5) * 2
        with torch.no_grad():
            x0 = img
            xs = []
            for it in range(self.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = int((self.t * len(self.betas)) // 1)
                a = (1 - self.betas).cumprod(dim=0).to(x0.device)
                x = (
                    x0 * a[total_noise_levels - 1].sqrt()
                    + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                )

                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * img.shape[0], device=img.device)
                    x = self.diffuse(x, t)
                xs.append(x)
            return (torch.cat(xs, dim=0) + 1) / 2


class Diffusion(BaseDiffusion):
    def __init__(
        self,
        model_path="google/ddpm-celebahq-256",
        num_diffusion_timesteps=1000,
        sample_step=1,
        t=0.2,
        image_size=256,
        device="cuda",
    ):
        betas = self.get_beta_schedule(
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )

        super().__init__(
            UNet2DModel.from_pretrained(model_path)
            .eval()
            .to(device)
            .requires_grad_(False),
            torch.from_numpy(betas).float().to(device),
            sample_step=sample_step,
            t=t,
            device=device,
        )

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def diffuse(self, x, t):
        alphas = 1.0 - self.betas.to(t.device)
        alphas_cumprod = alphas.cumprod(dim=0)

        model_output = self.model(x, t)["sample"]
        weighted_score = self.betas.to(t.device) / torch.sqrt(1 - alphas_cumprod)
        mean = self.extract(1 / torch.sqrt(alphas), t, x.shape) * (
            x - self.extract(weighted_score, t, x.shape) * model_output
        )

        logvar = self.extract(self.logvar, t, x.shape)
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
        sample = (
            mean
            + mask
            * torch.tensor(self.logvar, dtype=torch.float, device=mask.device)
            * noise
        )  # np.exp(0.5 * self.logvar) * noise
        return sample.float()

    @staticmethod
    def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    @staticmethod
    def extract(a, t, x_shape):
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(
            torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long()
        )
        assert out.shape == (bs,)
        out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
        return out


class GuidedDiffusion(BaseDiffusion):
    def __init__(
        self,
        model_path,
        num_diffusion_timesteps=1000,
        sample_step=1,
        t=0.2,
        image_size=256,
        device="cuda",
    ):
        model_config = dict(
            image_size=image_size,
            num_channels=image_size,
            num_res_blocks=2,
            num_heads=4,
            num_heads_upsample=-1,
            num_head_channels=64,
            attention_resolutions="32,16,8",
            channel_mult="",
            dropout=0.0,
            class_cond=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=True,
            use_new_attention_order=False,
            learn_sigma=True,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing=str(num_diffusion_timesteps),
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=False,
        )

        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.requires_grad_(False).eval().to(device)

        if model_config["use_fp16"]:
            model.convert_to_fp16()

        super().__init__(
            model,
            torch.from_numpy(diffusion.betas).float().to(device),
            sample_step=sample_step,
            t=t,
            device=device,
        )
        self.diffusion = diffusion

    def diffuse(self, x, t):
        return self.diffusion.p_sample(
            self.model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
        )["sample"]


class DiffusionAttack(BaseAttack):
    def __init__(
        self,
        evaluator,
        diffusion_type,
        diffusion_args,
        input_dir,
        output_dir,
        crop_ratio=None,
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
        assert diffusion_type in ["Diffusion", "GuidedDiffusion"]
        self.image_size = self.evaluator.image_size
        self.effective_size = next_power_of_2(self.image_size)
        self.optimizer = eval(diffusion_type)(
            image_size=256, device=device, **diffusion_args
        ).to(device)

        self.upscaler = transforms.Resize(
            (self.effective_size, self.effective_size), antialias=None
        )
        self.downscaler = transforms.Resize(
            (self.image_size, self.image_size), antialias=None
        )

        crop_ratio = (crop_ratio, crop_ratio) if crop_ratio is not None else None

        if crop_ratio is not None:
            crop_size = (
                int(crop_ratio[0] * self.image_size),
                int(crop_ratio[1] * self.image_size),
            )
            crop_layer = transforms.CenterCrop(crop_size)
            rescale_layer = transforms.Resize(
                (self.image_size, self.image_size), antialias=None
            )
            self.crop = transforms.Compose([crop_layer, rescale_layer])
        else:
            self.crop = lambda x: x

    def calc_sim_loss(self, removed, watermarked):
        wmd = self.crop(watermarked)
        return super().calc_sim_loss(removed, wmd)

    def do_batch(self, x):
        x = self.crop(x)
        x = self.upscaler(x)
        removed = self.optimizer(x).view(
            -1, 3, self.effective_size, self.effective_size
        )
        removed = self.downscaler(removed)
        return None, removed
