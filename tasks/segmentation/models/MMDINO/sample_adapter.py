import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MMDINO.sample_blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class SampleAdapter(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=[256, 512, 1024, 1024],
                 num_modalities: int = 1,
                 use_naf: bool = False,
                 naf_checkpoint: str = None,
                 naf_guidance_size: int = 224,
                 naf_backend: str = "cutlass-fna",
                 naf_q_tile_shape=None,
                 naf_kv_tile_shape=None):
        super(SampleAdapter, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=4,
                               stride=4,
                               padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=2,
                               stride=2,
                               padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3],
                      out_channels=out_channels[3],
                      kernel_size=3,
                      stride=2,
                      padding=1)
        ])

        self.num_modalities = num_modalities
        self.use_naf = use_naf
        self.naf = None
        self.naf_zero_conv = None
        self.naf_guidance_size = naf_guidance_size

        if self.use_naf:
            if num_modalities <= 1:
                raise ValueError("NAF P2 is only defined for multimodal fusion")
            if out_channels[0] != 256:
                raise ValueError(
                    "Released NAF P2 expects 256 scale-0 projection channels"
                )
            if naf_checkpoint is None:
                raise ValueError("naf_checkpoint is required when use_naf=True")
            if naf_guidance_size <= 0:
                raise ValueError("naf_guidance_size must be positive")

            # Lazy import keeps the faithful baseline independent of NATTEN.
            from .naf import load_released_naf

            self.naf = load_released_naf(
                naf_checkpoint,
                backend=naf_backend,
                q_tile_shape=naf_q_tile_shape,
                kv_tile_shape=naf_kv_tile_shape,
            )
            self.naf_zero_conv = nn.Conv2d(256,
                                           256,
                                           kernel_size=1,
                                           bias=False)
            nn.init.zeros_(self.naf_zero_conv.weight)

        if num_modalities > 1:
            self.modality_weights = nn.ParameterDict()
            for i in range(num_modalities):
                param_name = f'weight_modality_{i}'
                if param_name not in self.modality_weights:
                    self.modality_weights[param_name] = nn.Parameter(
                        torch.FloatTensor(1), requires_grad=True)
                    # 初始化权重，使它们的和接近1
                    self.modality_weights[param_name].data.fill_(
                        1.0 / num_modalities)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.naf is not None:
            # Outer model.train() must not enable NAF's stochastic RoPE mode.
            self.naf.eval()
        return self

    def forward(self,
                *features_list,
                patch_h=None,
                patch_w=None,
                guidance=None):
        if len(features_list) == 0:
            raise ValueError("At least one feature set must be provided")

        if len(features_list) == 1:
            # 单模态情况
            out = []
            for i, x in enumerate(features_list[0]):
                x = x.permute(0, 2, 1).reshape(
                    (x.shape[0], x.shape[-1], patch_h, patch_w))

                x = self.projects[i](x)
                x = self.resize_layers[i](x)

                out.append(x)

            return out
        else:
            # 多模态情况（2到n个模态）
            # 动态创建或获取模态权重参数
            if len(features_list) != self.num_modalities:
                raise ValueError(
                    f"Number of modalities ({len(features_list)}) does not match the number of modality weights ({self.num_modalities})"
                )
            num_modalities = len(features_list)

            naf_guidance = None
            if self.use_naf:
                if guidance is None:
                    raise ValueError("Optical guidance is required for NAF P2")
                if guidance.ndim != 4 or guidance.shape[1] != 3:
                    raise ValueError(
                        "NAF optical guidance must have shape [B, 3, H, W]"
                    )
                naf_guidance = F.interpolate(
                    guidance.detach(),
                    size=(self.naf_guidance_size, self.naf_guidance_size),
                    mode="bilinear",
                    align_corners=False,
                )

            # 处理每个模态的特征
            all_processed_features = [[] for _ in range(len(features_list))
                                      ]  # 为每个层级创建列表
            for i, modality_features in enumerate(zip(*features_list)):
                processed_features = []
                projected_features = [] if self.use_naf and i == 0 else None
                for feat in modality_features:
                    feat = feat.permute(0, 2, 1).reshape(
                        (feat.shape[0], feat.shape[-1], patch_h, patch_w))

                    feat = self.projects[i](feat)
                    if projected_features is not None:
                        projected_features.append(feat)
                    feat = self.resize_layers[i](feat)

                    processed_features.append(feat)

                naf_correction = None
                for j, feat in enumerate(processed_features):
                    weight_param_names = [
                        f'weight_modality_{i}' for i in range(num_modalities)
                    ]
                    weights = [
                        torch.sigmoid(self.modality_weights[name])
                        for name in weight_param_names
                    ]

                    # 归一化权重，确保它们的和为1
                    total_weight = sum(weights)
                    normalized_weights = [w / total_weight for w in weights]

                    # 加权求和
                    fused_feature = sum(w * f for w, f in zip(
                        normalized_weights, processed_features))

                    if self.use_naf and i == 0:
                        if naf_correction is None:
                            fused_low_resolution = sum(
                                w * f for w, f in zip(
                                    normalized_weights,
                                    projected_features,
                                ))
                            naf_upsampled = self.naf(
                                naf_guidance,
                                fused_low_resolution,
                                fused_feature.shape[-2:],
                            )
                            bilinear_upsampled = F.interpolate(
                                fused_low_resolution,
                                size=fused_feature.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            naf_delta = naf_upsampled - bilinear_upsampled
                            naf_correction = self.naf_zero_conv(naf_delta)
                        fused_feature = fused_feature + naf_correction

                    all_processed_features[j].append(fused_feature)

            return all_processed_features


class DPTHead(nn.Module):

    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=4,
                               stride=4,
                               padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=2,
                               stride=2,
                               padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3],
                      out_channels=out_channels[3],
                      kernel_size=3,
                      stride=2,
                      padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0))

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4,
                                         layer_3_rn,
                                         size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3,
                                         layer_2_rn,
                                         size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPT(nn.Module):

    def __init__(self,
                 encoder_size='base',
                 nclass=21,
                 features=128,
                 out_channels=[96, 192, 384, 768],
                 use_bn=False,
                 backbone=None):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }

        self.encoder_size = encoder_size
        self.backbone = backbone
        # Important: we freeze the backbone
        self.backbone.requires_grad_(False)
        self.head = DPTHead(nclass,
                            self.backbone.embed_dim,
                            features,
                            use_bn,
                            out_channels=out_channels)
        # self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # features = self.backbone.get_intermediate_layers(
        #     x, n = self.intermediate_layer_idx[self.encoder_size], reshape=True, norm=True
        # )
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size])

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16),
                            mode='bilinear',
                            align_corners=True)
        return out
