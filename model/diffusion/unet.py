"""
UNet implementation. Minorly modified from Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py

Set `smaller_encoder` to False for using larger observation encoder in ResidualBlock1D

"""

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import logging
from copy import deepcopy

log = logging.getLogger(__name__)

from model.diffusion.modules import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from model.common.mlp import ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

class ResidualBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=5,
        n_groups=None,
        cond_predict_scale=False,
        larger_encoder=False,
        activation_type="Mish",
        groupnorm_eps=1e-5,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                    activation_type=activation_type,
                    eps=groupnorm_eps,
                ),
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                    activation_type=activation_type,
                    eps=groupnorm_eps,
                ),
            ]
        )
        if activation_type == "Mish":
            act = nn.Mish()
        elif activation_type == "ReLU":
            act = nn.ReLU()
        else:
            raise "Unknown activation type for ConditionalResidualBlock1D"

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        if larger_encoder:
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, cond_channels),
                act,
                nn.Linear(cond_channels, cond_channels),
                act,
                nn.Linear(cond_channels, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            self.cond_encoder = nn.Sequential(
                act,
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon_steps ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon_steps ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Unet1D(nn.Module):

    def __init__(
        self,
        action_dim,
        cond_dim=None,
        diffusion_step_embed_dim=32,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        smaller_encoder=False,
        cond_mlp_dims=None,
        kernel_size=5,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        groupnorm_eps=1e-5,
    ):
        super().__init__()
        dims = [action_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        log.info(f"Channel dimensions: {in_out}")

        dsed = diffusion_step_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        if cond_mlp_dims is not None:
            self.cond_mlp = ResidualMLP(
                dim_list=[cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            cond_block_dim = dsed + cond_mlp_dims[-1]
        else:
            cond_block_dim = dsed + cond_dim
        use_large_encoder_in_block = cond_mlp_dims is None and not smaller_encoder

        mid_dim = dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
                ResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        ResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        ResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(
                dim,
                dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                activation_type=activation_type,
                eps=groupnorm_eps,
            ),
            nn.Conv1d(dim, action_dim, 1),
        )

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, act_dim)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, obs_dim)
        """
        B = len(x)

        # move chunk dim to the end
        x = einops.rearrange(x, "b h t -> b t h")

        # flatten history
        state = cond["state"].view(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # 1. time
        if not torch.is_tensor(time):
            time = torch.tensor([time], dtype=torch.long, device=x.device)
        elif torch.is_tensor(time) and len(time.shape) == 0:
            time = time[None].to(x.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        time = time.expand(x.shape[0])
        global_feature = self.time_mlp(time)
        global_feature = torch.cat([global_feature, state], axis=-1)

        # encode local features
        h_local = list()
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


class VisionUnet1D(nn.Module):

    def __init__(
        self,
        backbone,
        action_dim,
        img_cond_steps=1,
        cond_dim=None,
        diffusion_step_embed_dim=32,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        smaller_encoder=False,
        cond_mlp_dims=None,
        kernel_size=5,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        groupnorm_eps=1e-5,
        spatial_emb=0,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        # unet
        dims = [action_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        log.info(f"Channel dimensions: {in_out}")

        dsed = diffusion_step_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        if cond_mlp_dims is not None:
            self.cond_mlp = ResidualMLP(
                dim_list=[cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            cond_block_dim = dsed + cond_mlp_dims[-1] + visual_feature_dim
        else:
            cond_block_dim = dsed + cond_dim + visual_feature_dim
        use_large_encoder_in_block = cond_mlp_dims is None and not smaller_encoder

        mid_dim = dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
                ResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        ResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        ResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_block_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder_in_block,
                            activation_type=activation_type,
                            groupnorm_eps=groupnorm_eps,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(
                dim,
                dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                activation_type=activation_type,
                eps=groupnorm_eps,
            ),
            nn.Conv1d(dim, action_dim, 1),
        )

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, act_dim)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, obs_dim)
        """
        B = len(x)
        _, T_rgb, C, H, W = cond["rgb"].shape

        # move chunk dim to the end
        x = einops.rearrange(x, "b h t -> b t h")

        # flatten history
        state = cond["state"].view(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        cond_encoded = torch.cat([feat, state], dim=-1)

        # 1. time
        if not torch.is_tensor(time):
            time = torch.tensor([time], dtype=torch.long, device=x.device)
        elif torch.is_tensor(time) and len(time.shape) == 0:
            time = time[None].to(x.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        time = time.expand(x.shape[0])
        global_feature = self.time_mlp(time)
        global_feature = torch.cat([global_feature, cond_encoded], axis=-1)

        # encode local features
        h_local = list()
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x

