import re
from typing import Any
import pickle
import torch
import numpy as np


class _TFNetworkStub:
    pass


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def populate_module_params(module, *patterns):
    for name, tensor in named_params_and_buffers(module):
        found = False
        value = None
        for pattern, value_fn in zip(patterns[0::2], patterns[1::2]):
            match = re.fullmatch(pattern, name)
            if match:
                found = True
                if value_fn is not None:
                    value = value_fn(*match.groups())
                break
        try:
            assert found
            if value is not None:
                tensor.copy_(torch.from_numpy(np.array(value)))
        except:
            print(name, list(tensor.shape))
            raise


def decoder_patterns(tf_params):
    return [
        r"b(\d+)\.fromrgb\.weight",
        lambda r: tf_params[f"{r}x{r}/FromRGB/weight"].transpose(3, 2, 0, 1),
        r"b(\d+)\.fromrgb\.bias",
        lambda r: tf_params[f"{r}x{r}/FromRGB/bias"],
        r"b(\d+)\.conv(\d+)\.weight",
        lambda r, i: tf_params[
            f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/weight'
        ].transpose(3, 2, 0, 1),
        r"b(\d+)\.conv(\d+)\.bias",
        lambda r, i: tf_params[f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/bias'],
        r"b(\d+)\.skip\.weight",
        lambda r: tf_params[f"{r}x{r}/Skip/weight"].transpose(3, 2, 0, 1),
        r"mapping\.embed\.weight",
        lambda: tf_params[f"LabelEmbed/weight"].transpose(),
        r"mapping\.embed\.bias",
        lambda: tf_params[f"LabelEmbed/bias"],
        r"mapping\.fc(\d+)\.weight",
        lambda i: tf_params[f"Mapping{i}/weight"].transpose(),
        r"mapping\.fc(\d+)\.bias",
        lambda i: tf_params[f"Mapping{i}/bias"],
        r"b4\.conv\.weight",
        lambda: tf_params[f"4x4/Conv/weight"].transpose(3, 2, 0, 1),
        r"b4\.conv\.bias",
        lambda: tf_params[f"4x4/Conv/bias"],
        r"b4\.fc\.weight",
        lambda: tf_params[f"4x4/Dense0/weight"].transpose(),
        r"b4\.fc\.bias",
        lambda: tf_params[f"4x4/Dense0/bias"],
        r"b4\.out\.weight",
        lambda: tf_params[f"Output/weight"].transpose(),
        r"b4\.out\.bias",
        lambda: tf_params[f"Output/bias"],
        r".*\.resample_filter",
        None,
    ]


def encoder_watermark_patterns(tf_params):
    return [
        r"mapping\.w_avg",
        lambda: tf_params[f"dlatent_avg"],
        r"mapping\.embed\.weight",
        lambda: tf_params[f"mapping/LabelEmbed/weight"].transpose(),
        r"mapping\.embed\.bias",
        lambda: tf_params[f"mapping/LabelEmbed/bias"],
        r"mapping\.fc(\d+)\.weight",
        lambda i: tf_params[f"mapping/Dense{i}/weight"].transpose(),
        r"mapping\.fc(\d+)\.bias",
        lambda i: tf_params[f"mapping/Dense{i}/bias"],
        r"synthesis\.b4\.conv0\.weight",
        lambda: tf_params[f"synthesis/4x4/Latent/weight"].transpose(),
        r"synthesis\.b4\.conv0\.bias",
        lambda: tf_params[f"synthesis/4x4/Latent/bias"],
        r"synthesis\.b4\.conv1\.weight",
        lambda: tf_params[f"synthesis/4x4/Conv/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b4\.conv1\.bias",
        lambda: tf_params[f"synthesis/4x4/Conv/bias"],
        r"synthesis\.b4\.conv1\.noise_const",
        lambda: tf_params[f"synthesis/noise0"][0, 0],
        r"synthesis\.b4\.conv1\.noise_strength",
        lambda: tf_params[f"synthesis/4x4/Conv/noise_strength"],
        r"synthesis\.b4\.conv1\.affine\.weight",
        lambda: tf_params[f"synthesis/4x4/Conv/mod_weight"].transpose(),
        r"synthesis\.b4\.conv1\.affine\.bias",
        lambda: tf_params[f"synthesis/4x4/Conv/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.conv0\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/weight"][::-1, ::-1].transpose(
            3, 2, 0, 1
        ),
        r"synthesis\.b(\d+)\.conv0\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/bias"],
        r"synthesis\.b(\d+)\.conv0\.noise_const",
        lambda r: tf_params[f"synthesis/noise{int(np.log2(int(r)))*2-5}"][0, 0],
        r"synthesis\.b(\d+)\.conv0\.noise_strength",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/noise_strength"],
        r"synthesis\.b(\d+)\.conv0\.affine\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.conv0\.affine\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.conv1\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.conv1\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/bias"],
        r"synthesis\.b(\d+)\.conv1\.noise_const",
        lambda r: tf_params[f"synthesis/noise{int(np.log2(int(r)))*2-4}"][0, 0],
        r"synthesis\.b(\d+)\.conv1\.noise_strength",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/noise_strength"],
        r"synthesis\.b(\d+)\.conv1\.affine\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.conv1\.affine\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.torgb\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.torgb\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/bias"],
        r"synthesis\.b(\d+)\.torgb\.affine\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.torgb\.affine\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.skip\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Skip/weight"][::-1, ::-1].transpose(
            3, 2, 0, 1
        ),
        r".*\.resample_filter",
        None,
    ]


def encoder_patterns(tf_params):
    return [
        r"mapping\.w_avg",
        lambda: tf_params[f"dlatent_avg"],
        r"mapping\.embed\.weight",
        lambda: tf_params[f"mapping/LabelEmbed/weight"].transpose(),
        r"mapping\.embed\.bias",
        lambda: tf_params[f"mapping/LabelEmbed/bias"],
        r"mapping\.fc(\d+)\.weight",
        lambda i: tf_params[f"mapping/Dense{i}/weight"].transpose(),
        r"mapping\.fc(\d+)\.bias",
        lambda i: tf_params[f"mapping/Dense{i}/bias"],
        r"synthesis\.b4\.const",
        lambda: tf_params[f"synthesis/4x4/Const/const"][0],
        r"synthesis\.b4\.conv1\.weight",
        lambda: tf_params[f"synthesis/4x4/Conv/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b4\.conv1\.bias",
        lambda: tf_params[f"synthesis/4x4/Conv/bias"],
        r"synthesis\.b4\.conv1\.noise_const",
        lambda: tf_params[f"synthesis/noise0"][0, 0],
        r"synthesis\.b4\.conv1\.noise_strength",
        lambda: tf_params[f"synthesis/4x4/Conv/noise_strength"],
        r"synthesis\.b4\.conv1\.affine\.weight",
        lambda: tf_params[f"synthesis/4x4/Conv/mod_weight"].transpose(),
        r"synthesis\.b4\.conv1\.affine\.bias",
        lambda: tf_params[f"synthesis/4x4/Conv/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.conv0\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/weight"][::-1, ::-1].transpose(
            3, 2, 0, 1
        ),
        r"synthesis\.b(\d+)\.conv0\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/bias"],
        r"synthesis\.b(\d+)\.conv0\.noise_const",
        lambda r: tf_params[f"synthesis/noise{int(np.log2(int(r)))*2-5}"][0, 0],
        r"synthesis\.b(\d+)\.conv0\.noise_strength",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/noise_strength"],
        r"synthesis\.b(\d+)\.conv0\.affine\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.conv0\.affine\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv0_up/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.conv1\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.conv1\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/bias"],
        r"synthesis\.b(\d+)\.conv1\.noise_const",
        lambda r: tf_params[f"synthesis/noise{int(np.log2(int(r)))*2-4}"][0, 0],
        r"synthesis\.b(\d+)\.conv1\.noise_strength",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/noise_strength"],
        r"synthesis\.b(\d+)\.conv1\.affine\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.conv1\.affine\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/Conv1/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.torgb\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.torgb\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/bias"],
        r"synthesis\.b(\d+)\.torgb\.affine\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.torgb\.affine\.bias",
        lambda r: tf_params[f"synthesis/{r}x{r}/ToRGB/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.skip\.weight",
        lambda r: tf_params[f"synthesis/{r}x{r}/Skip/weight"][::-1, ::-1].transpose(
            3, 2, 0, 1
        ),
        r".*\.resample_filter",
        None,
    ]


def decoder_kwargs(tf_kwargs):
    known_kwargs = set()

    def kwarg(tf_name, default=None):
        known_kwargs.add(tf_name)
        return tf_kwargs.get(tf_name, default)

    kwargs = EasyDict(
        watermark_dim=kwarg("watermark_size", 128),
        z_dim=kwarg("latent_size", 512),
        c_dim=kwarg("label_size", 0),
        img_resolution=kwarg("resolution", 1024),
        img_channels=kwarg("num_channels", 3),
        architecture=kwarg("architecture", "resnet"),
        channel_base=kwarg("fmap_base", 16384) * 2,
        channel_max=kwarg("fmap_max", 512),
        num_fp16_res=kwarg("num_fp16_res", 0),
        conv_clamp=kwarg("conv_clamp", None),
        cmap_dim=kwarg("mapping_fmaps", None),
        block_kwargs=EasyDict(
            activation=kwarg("nonlinearity", "lrelu"),
            resample_filter=kwarg("resample_kernel", [1, 3, 3, 1]),
            freeze_layers=kwarg("freeze_layers", 0),
        ),
        mapping_kwargs=EasyDict(
            num_layers=kwarg("mapping_layers", 0),
            embed_features=kwarg("mapping_fmaps", None),
            layer_features=kwarg("mapping_fmaps", None),
            activation=kwarg("nonlinearity", "lrelu"),
            lr_multiplier=kwarg("mapping_lrmul", 0.1),
        ),
        epilogue_kwargs=EasyDict(
            mbstd_group_size=kwarg("mbstd_group_size", None),
            mbstd_num_channels=kwarg("mbstd_num_features", 1),
            activation=kwarg("nonlinearity", "lrelu"),
        ),
    )

    # Check for unknown kwargs.
    kwarg("structure")
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError("Unknown TensorFlow kwarg", unknown_kwargs[0])
    return kwargs


def encoder_watermark_kwargs(tf_kwargs):
    known_kwargs = set()

    def kwarg(tf_name, default=None, none=None):
        known_kwargs.add(tf_name)
        val = tf_kwargs.get(tf_name, default)
        return val if val is not None else none

    # Convert kwargs.
    kwargs = EasyDict(
        z_dim=kwarg("latent_size", 512),
        c_dim=kwarg("label_size", 0),
        w_dim=kwarg("dlatent_size", 512),
        water_dim=kwarg("watermark_size", 128),
        res_modulated_max_log2=kwarg("res_modulated_max_log2", 7),
        res_modulated_min_log2=kwarg("res_modulated_min_log2", 2),
        img_resolution=kwarg("resolution", 1024),
        img_channels=kwarg("num_channels", 3),
        mapping_kwargs=EasyDict(
            num_layers=kwarg("mapping_layers", 8),
            embed_features=kwarg("label_fmaps", None),
            layer_features=kwarg("mapping_fmaps", None),
            activation=kwarg("mapping_nonlinearity", "lrelu"),
            lr_multiplier=kwarg("mapping_lrmul", 0.01),
            w_avg_beta=kwarg("w_avg_beta", 0.995, none=1),
        ),
        synthesis_kwargs=EasyDict(
            channel_base=kwarg("fmap_base", 16384) * 2,
            channel_max=kwarg("fmap_max", 512),
            num_fp16_res=kwarg("num_fp16_res", 0),
            conv_clamp=kwarg("conv_clamp", None),
            architecture=kwarg("architecture", "skip"),
            resample_filter=kwarg("resample_kernel", [1, 3, 3, 1]),
            use_noise=kwarg("use_noise", True),
            activation=kwarg("nonlinearity", "lrelu"),
        ),
    )

    # Check for unknown kwargs.
    kwarg("truncation_psi")
    kwarg("truncation_cutoff")
    kwarg("style_mixing_prob")
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError("Unknown TensorFlow kwarg", unknown_kwargs[0])
    return kwargs


def encoder_kwargs(tf_kwargs):
    known_kwargs = set()

    def kwarg(tf_name, default=None, none=None):
        known_kwargs.add(tf_name)
        val = tf_kwargs.get(tf_name, default)
        return val if val is not None else none

    # Convert kwargs.
    kwargs = EasyDict(
        z_dim=kwarg("latent_size", 512),
        c_dim=kwarg("label_size", 0),
        w_dim=kwarg("dlatent_size", 512),
        img_resolution=kwarg("resolution", 1024),
        img_channels=kwarg("num_channels", 3),
        mapping_kwargs=EasyDict(
            num_layers=kwarg("mapping_layers", 8),
            embed_features=kwarg("label_fmaps", None),
            layer_features=kwarg("mapping_fmaps", None),
            activation=kwarg("mapping_nonlinearity", "lrelu"),
            lr_multiplier=kwarg("mapping_lrmul", 0.01),
            w_avg_beta=kwarg("w_avg_beta", 0.995, none=1),
        ),
        synthesis_kwargs=EasyDict(
            channel_base=kwarg("fmap_base", 16384) * 2,
            channel_max=kwarg("fmap_max", 512),
            num_fp16_res=kwarg("num_fp16_res", 0),
            conv_clamp=kwarg("conv_clamp", None),
            architecture=kwarg("architecture", "skip"),
            resample_filter=kwarg("resample_kernel", [1, 3, 3, 1]),
            use_noise=kwarg("use_noise", True),
            activation=kwarg("nonlinearity", "lrelu"),
        ),
    )

    # Check for unknown kwargs.
    kwarg("truncation_psi")
    kwarg("truncation_cutoff")
    kwarg("style_mixing_prob")
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError("Unknown TensorFlow kwarg", unknown_kwargs[0])
    return kwargs


def extract_kwargs(tf_C, c_type):
    if tf_C.version < 4:
        raise ValueError("TensorFlow pickle version too low")
    tf_kwargs = tf_C.static_kwargs

    if c_type == "Decoder":
        kwargs = decoder_kwargs(tf_kwargs)
    elif c_type == "WatermarkEnc":
        kwargs = encoder_watermark_kwargs(tf_kwargs)
    else:
        kwargs = encoder_kwargs(tf_kwargs)
    return kwargs


def extract_patterns(tf_params, c_type):
    if c_type == "Decoder":
        return decoder_patterns(tf_params)
    if c_type == "WatermarkEnc":
        return encoder_watermark_patterns(tf_params)
    return encoder_patterns(tf_params)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "dnnlib.tflib.optimizer" or (
            module == "dnnlib.tflib.network" and name == "Network"
        ):
            return _TFNetworkStub
        return super().find_class(module, name)


def _collect_tf_params(tf_net):
    tf_params = dict()

    def recurse(prefix, tf_net):
        for name, value in tf_net.variables:
            tf_params[prefix + name] = value
        for name, comp in tf_net.components.items():
            recurse(prefix + name + "/", comp)

    recurse("", tf_net)
    return tf_params


def extract_tf_params(tf_C):
    tf_params = _collect_tf_params(tf_C)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r"ToRGB_lod(\d+)/(.*)", name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f"{r}x{r}/ToRGB/{match.group(2)}"] = value
            kwargs.synthesis.kwargs.architecture = "orig"
    return tf_params


def load_tf_pickle(network_pkl, c_type):
    idx = 0 if c_type == "Decoder" else (3 if c_type == "WatermarkEnc" else 2)
    with open(network_pkl, "rb") as stream:
        Gs = _LegacyUnpickler(stream).load()[idx]
    return Gs


def load_tf_network(network_pkl, c_type):
    assert c_type in ["Decoder", "Encoder", "WatermarkEnc"]
    tf_C = load_tf_pickle(network_pkl, c_type)
    kwargs = extract_kwargs(tf_C, c_type)
    tf_params = extract_tf_params(tf_C)
    patterns = extract_patterns(tf_params, c_type)
    return (kwargs, patterns)
