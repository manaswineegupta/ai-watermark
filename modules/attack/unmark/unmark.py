import os
import numpy as np
import torch
from torchvision import transforms
from modules.attack import BaseAttack
from .cw import SpecialCWCoordinate
from .losses import get_loss


class UnMark(BaseAttack):
    def __init__(
        self,
        evaluator,
        stage_selector,
        preprocess_args,
        stage1_args,
        stage2_args,
        input_dir,
        output_dir,
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
        stage_selector = (
            ["preprocess", "stage1", "stage2"]
            if stage_selector is None
            else stage_selector
        )
        all_args = [
            s_args if name in stage_selector else None
            for name, s_args in zip(
                ["preprocess", "stage1", "stage2"],
                (preprocess_args, stage1_args, stage2_args),
            )
        ]
        preprocess_args, stage1_args, stage2_args = all_args

        preprocess_args = {} if preprocess_args is None else preprocess_args
        crop_size = (
            (
                int(preprocess_args["crop_ratio"][0] * self.image_size),
                int(preprocess_args["crop_ratio"][1] * self.image_size),
            )
            if preprocess_args.get("crop_ratio") is not None
            else None
        )
        crop_layer = (
            transforms.CenterCrop(crop_size)
            if crop_size is not None
            else transforms.Lambda(lambda x: x)
        )
        rescale_layer = (
            transforms.Resize((self.image_size, self.image_size), antialias=None)
            if crop_size is not None
            else transforms.Lambda(lambda x: x)
        )
        self.transforms = transforms.Compose(
            [
                crop_layer,
                rescale_layer,
            ]
        )

        self.stage1, self.stage1_thresh = self._load_stage(
            stage1_args, stage_name="high_freq"
        )
        self.stage2, self.stage2_thresh = self._load_stage(
            stage2_args, stage_name="low_freq"
        )

    def calc_sim_loss(self, removed, watermarked):
        wmd = self.transforms(watermarked)
        return super().calc_sim_loss(removed, wmd)

    def _load_stage(self, stage_args, stage_name):
        assert stage_args is None or isinstance(stage_args, dict)
        if isinstance(stage_args, dict):
            for name in ["loss_fn", "dist_fn", "loss_thresh", "optimizer_args"]:
                assert name in list(stage_args.keys())
        if stage_args is None:
            stage = lambda x, y, ox: x
            return stage, {}
        stage_loss_args = stage_args["loss_fn"].get("args", {})
        stage_loss_args = stage_loss_args if stage_loss_args is not None else {}
        stage_dist_args = stage_args["dist_fn"].get("args", {})
        stage_dist_args = stage_dist_args if stage_dist_args is not None else {}
        stage_loss = get_loss(stage_args["loss_fn"]["type"], **stage_loss_args).to(
            self.device
        )
        stage_dist = get_loss(stage_args["dist_fn"]["type"], **stage_dist_args).to(
            self.device
        )
        stage_thresh = float(stage_args["loss_thresh"])
        for name in ["loss_fn", "dist_fn", "loss_thresh"]:
            del stage_args[name]

        if stage_args.get("progress_bar_args") is not None:
            stage_args["progress_bar_args"]["stage_name"] = stage_name

        stage = SpecialCWCoordinate(
            stage_loss,
            stage_dist,
            self.evaluator,
            device=self.device,
            **stage_args,
        )
        return stage, {"thresh": stage_thresh}

    def do_batch(self, x):
        stage0 = self.transforms(x)
        stage1 = self.stage1(stage0, stage0, ox=stage0, **self.stage1_thresh)
        removed = self.stage2(
            stage1,
            stage1,
            ox=stage1,
            **self.stage2_thresh,
        )
        return None, removed
