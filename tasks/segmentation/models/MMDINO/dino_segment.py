import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_decoder import LinearHead
from .Decoder import Decoder, Decoder_FRM, Decoder_PRN, Decoder_MMFF, Decoder_FRM_MMFF, Decoder_PRN_MMFF, Decoder_FRM_PRN
from .sample_adapter import SampleAdapter
from .availability import active_modality_indices
from .lora import LoRA
from .ResNet import ResNet50

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dinov3.hub.backbones import dinov3_vitl16, dinov3_vits16plus, dinov3_vitb16, dinov3_vits16, dinov3_vit7b16

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vits16plus": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class DINOSegmentModule(nn.Module):

    def __init__(
        self,
        backbone_weights=None,
        freeze_backbone: bool = False,
        n_classes: int = 1000,
        # window_size=(224, 224),
        use_lora: bool = False,
        r: int = 3,
        decoder_type='Decoder',
        adapter_type=None,
        backbone_type='dinov3_vitl16',
        # lora_layers=None,
        num_modalities: int = 1,
        use_naf: bool = False,
        naf_checkpoint: str = None,
        naf_guidance_size: int = 224,
        naf_backend: str = "cutlass-fna",
        naf_q_tile_shape=None,
        naf_kv_tile_shape=None,
        use_optical_stem: bool = False,
        optical_stem_seed: int = 0,
        raw_logits: bool = False,
    ):
        super().__init__()

        self.num_modalities = num_modalities
        self.use_optical_stem = bool(use_optical_stem)
        if self.use_optical_stem and decoder_type != 'Decoder':
            raise ValueError(
                "the optical spatial stem is only implemented for Decoder"
            )

        dinov3_vits_dict = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vits16plus": dinov3_vits16plus,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
            "dinov3_vit7b16": dinov3_vit7b16
        }
        dinov3_vit = dinov3_vits_dict[backbone_type]
        self.backbone_type = backbone_type
        if backbone_weights is not None:
            self.backbone = dinov3_vit(weights=backbone_weights,
                                       pretrained=True)
        else:
            self.backbone = dinov3_vit(pretrained=False)

        # Important: we freeze the backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        embed_dim = self.backbone.embed_dim

        # 根据类型选择适配器
        self.adapter = None
        if adapter_type == 'SampleAdapter':
            self.adapter = SampleAdapter(embed_dim,
                                         num_modalities=num_modalities,
                                         use_naf=use_naf,
                                         naf_checkpoint=naf_checkpoint,
                                         naf_guidance_size=naf_guidance_size,
                                         naf_backend=naf_backend,
                                         naf_q_tile_shape=naf_q_tile_shape,
                                         naf_kv_tile_shape=naf_kv_tile_shape)

        # 根据类型选择解码器
        decoder_kwargs = {
            "n_classes": n_classes,
            "num_modalities": num_modalities
        }
        if adapter_type is None:
            decoder_kwargs["in_channels"] = [embed_dim] * 4
        if decoder_type != 'LinearHead':
            decoder_kwargs["raw_logits"] = raw_logits

        if decoder_type == 'LinearHead':
            self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        elif decoder_type == 'Decoder':
            decoder_kwargs["use_optical_stem"] = self.use_optical_stem
            decoder_kwargs["optical_stem_seed"] = optical_stem_seed
            self.decoder = Decoder(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM':
            self.decoder = Decoder_FRM(**decoder_kwargs)
        elif decoder_type == 'Decoder_MMFF':
            self.decoder = Decoder_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN':
            self.decoder = Decoder_PRN(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_MMFF':
            self.decoder = Decoder_FRM_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN_MMFF':
            self.decoder = Decoder_PRN_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_PRN':
            self.decoder = Decoder_FRM_PRN(**decoder_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    @staticmethod
    def _validate_modality_batch(modalities, *, require_same_spatial=False):
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")
        batch_size = modalities[0].shape[0]
        if any(modality.shape[0] != batch_size for modality in modalities):
            raise ValueError("All modality inputs must have the same batch size")
        if require_same_spatial:
            spatial_shape = modalities[0].shape[-2:]
            if any(modality.shape[-2:] != spatial_shape for modality in modalities):
                raise ValueError("All modality inputs must have the same spatial shape")
        return batch_size

    @staticmethod
    def _prepare_dino_modality(modality):
        if modality.shape[1] == 1:
            return modality.repeat(1, 3, 1, 1)
        if modality.shape[1] != 3:
            raise ValueError(
                "DINOv3 modality inputs must have one or three channels"
            )
        return modality

    def _extract_backbone_output(self, modality):
        modality = self._prepare_dino_modality(modality)
        return self.backbone.get_intermediate_layers(
            modality,
            n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type],
        )

    def extract_frozen_backbone_outputs(self, *modalities):
        """Extract one reusable raw DINO output tuple per canonical slot.

        Reusing these tensors across sequential backward calls is valid only
        while the backbone is fully frozen. Existing ``forward`` callers keep
        their original branch-skipping behavior and do not use this API.
        """

        self._validate_modality_batch(modalities, require_same_spatial=True)
        if len(modalities) != self.num_modalities:
            raise ValueError(
                "Frozen feature extraction requires one tensor for every "
                "canonical modality slot"
            )
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError(
                "Cached raw backbone outputs require a fully frozen backbone"
            )
        with torch.no_grad():
            return tuple(
                self._extract_backbone_output(modality) for modality in modalities
            )

    def _decode_active_backbone_outputs(
        self,
        modalities,
        outputs_modalities,
        *,
        active_indices,
        canonical_slots,
    ):
        if len(outputs_modalities) != len(active_indices):
            raise ValueError("Active modalities and backbone outputs do not match")

        x = modalities[active_indices[0]]
        _, _, height, width = x.shape
        if any(
            modalities[index].shape[-2:] != (height, width)
            for index in active_indices
        ):
            raise ValueError("All active modalities must have the same spatial shape")
        patch_h, patch_w = height // 16, width // 16
        scale_factors = [4, 2, 1, 0.5]

        if self.adapter is not None:
            processed_outputs_modalities = self.adapter(
                *outputs_modalities,
                patch_h=patch_h,
                patch_w=patch_w,
                guidance=modalities[0] if 0 in active_indices else None,
                modality_indices=active_indices if canonical_slots else None,
            )
        else:
            processed_outputs_modalities = []
            for outputs_modality in outputs_modalities:
                processed_outputs_modality = []
                for index, output in enumerate(outputs_modality):
                    output = output.permute(0, 2, 1).reshape(
                        (output.shape[0], output.shape[-1], patch_h, patch_w)
                    )
                    if index < len(scale_factors):
                        output = F.interpolate(
                            output,
                            scale_factor=scale_factors[index],
                            mode="bilinear",
                            align_corners=False,
                        )
                    processed_outputs_modality.append(output)
                processed_outputs_modalities.append(processed_outputs_modality)

        if self.use_optical_stem:
            logits = self.decoder(
                *processed_outputs_modalities,
                guidance=modalities[0],
            )
        else:
            logits = self.decoder(*processed_outputs_modalities)

        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
            )
        return logits

    def forward_from_backbone_outputs(
        self,
        *modalities,
        backbone_outputs,
        availability=None,
    ):
        """Decode one canonical availability state from cached raw outputs."""

        batch_size = self._validate_modality_batch(
            modalities,
            require_same_spatial=True,
        )
        if len(modalities) != self.num_modalities:
            raise ValueError(
                "Cached canonical forward requires one tensor for every modality slot"
            )
        if len(backbone_outputs) != len(modalities):
            raise ValueError(
                "Cached backbone output count must match the modality slot count"
            )
        active_indices = active_modality_indices(
            availability,
            batch_size=batch_size,
            num_modalities=self.num_modalities,
        )
        if self.use_optical_stem and (len(active_indices) == 1 or 0 not in active_indices):
            raise ValueError(
                "the optical spatial stem requires optical and multimodal inputs"
            )
        active_outputs = tuple(backbone_outputs[index] for index in active_indices)
        return self._decode_active_backbone_outputs(
            modalities,
            active_outputs,
            active_indices=active_indices,
            canonical_slots=availability is not None,
        )

    def forward(self, *modalities, availability=None):
        batch_size = self._validate_modality_batch(modalities)
        if availability is not None and len(modalities) != self.num_modalities:
            raise ValueError(
                "Canonical availability requires one tensor for every modality slot"
            )
        active_indices = active_modality_indices(
            availability,
            batch_size=batch_size,
            num_modalities=(
                len(modalities) if availability is None else self.num_modalities
            ),
        )
        canonical_slots = availability is not None
        if self.use_optical_stem and (len(active_indices) == 1 or 0 not in active_indices):
            raise ValueError(
                "the optical spatial stem requires optical and multimodal inputs"
            )

        # 主输入x
        x = modalities[active_indices[0]]
        _, _, H, W = x.shape
        if any(
            modalities[index].shape[-2:] != (H, W)
            for index in active_indices
        ):
            raise ValueError("All active modalities must have the same spatial shape")
        patch_h, patch_w = H // 16, W // 16

        scale_factors = [4, 2, 1, 0.5]

        if len(active_indices) == 1 and not canonical_slots:
            if self.adapter is not None:
                outputs = self.backbone.get_intermediate_layers(
                    x, n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type])
                # 使用适配器处理多尺度特征
                multi_scale_features = self.adapter(outputs,
                                                    patch_h=patch_h,
                                                    patch_w=patch_w)
            else:
                outputs = self.backbone.get_intermediate_layers(
                    x,
                    n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type],
                    reshape=True)
                # 直接处理中间层输出
                multi_scale_features = []
                for i, output in enumerate(outputs):
                    if i < len(scale_factors):
                        output = F.interpolate(output,
                                               scale_factor=scale_factors[i],
                                               mode="bilinear",
                                               align_corners=False)
                    multi_scale_features.append(output)

            logits = self.decoder(multi_scale_features)

        else:
            outputs_modalities = [
                self._extract_backbone_output(modalities[modality_index])
                for modality_index in active_indices
            ]

            if self.adapter is not None:
                # 使用适配器处理多尺度特征
                processed_outputs_modalities = self.adapter(
                    *outputs_modalities,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    guidance=modalities[0] if 0 in active_indices else None,
                    modality_indices=active_indices if canonical_slots else None)
            else:
                processed_outputs_modalities = []
                for outputs_modality in outputs_modalities:
                    # 直接处理中间层输出
                    processed_outputs_modality = []
                    for i, output in enumerate(outputs_modality):
                        output = output.permute(0, 2, 1).reshape(
                            (output.shape[0], output.shape[-1], patch_h,
                             patch_w))

                        if i < len(scale_factors):
                            output = F.interpolate(
                                output,
                                scale_factor=scale_factors[i],
                                mode="bilinear",
                                align_corners=False)
                        processed_outputs_modality.append(output)

                    processed_outputs_modalities.append(
                        processed_outputs_modality)

            # 将处理后的所有模态特征传递给解码器
            if self.use_optical_stem:
                logits = self.decoder(*processed_outputs_modalities,
                                      guidance=modalities[0])
            else:
                logits = self.decoder(*processed_outputs_modalities)

        _H, _W = logits.shape[2:]
        if _H != H or _W != W:
            # 确保输出大小与输入一致
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
            )

        return logits


class ResNetSegmentModule(nn.Module):

    def __init__(
        self,
        n_classes: int = 1000,
        use_lora: bool = False,
        r: int = 3,
        decoder_type='Decoder',
        num_modalities: int = 1,
    ):
        super().__init__()

        self.backbone = ResNet50(pretrained=True)

        embed_dim = [256, 512, 1024, 2048]

        # 根据类型选择解码器
        decoder_kwargs = {
            "in_channels": embed_dim,
            "n_classes": n_classes,
            "num_modalities": num_modalities
        }

        if decoder_type == 'LinearHead':
            self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        elif decoder_type == 'Decoder':
            self.decoder = Decoder(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM':
            self.decoder = Decoder_FRM(**decoder_kwargs)
        elif decoder_type == 'Decoder_MMFF':
            self.decoder = Decoder_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN':
            self.decoder = Decoder_PRN(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_MMFF':
            self.decoder = Decoder_FRM_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN_MMFF':
            self.decoder = Decoder_PRN_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_PRN':
            self.decoder = Decoder_FRM_PRN(**decoder_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, *modalities):
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # 主输入x
        x = modalities[0]
        _, C, H, W = x.shape

        if len(modalities) == 1:
            outputs = self.backbone(x)

            logits = self.decoder(outputs)

        else:
            outputs_modalities = []
            for idx, modality_input in enumerate(modalities):
                if modality_input.shape[1] != C and idx > 0:
                    modality_input = modality_input.repeat(1, C, 1, 1)

                outputs_modality = self.backbone(modality_input)
                outputs_modalities.append(outputs_modality)

            # 将处理后的所有模态特征传递给解码器
            logits = self.decoder(*outputs_modalities)

        _H, _W = logits.shape[2:]
        if _H != H or _W != W:
            # 确保输出大小与输入一致
            pred = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
            )

        return pred


def build_model(
    model_name=None,
    backbone_weights=None,
    n_classes: int = 1000,
    use_lora: bool = False,
    r: int = 3,
    num_modalities: int = 1,
    **kwargs,
):
    if model_name == 'DINOv3' or model_name is None:
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            num_modalities=num_modalities,
            adapter_type="SampleAdapter",
            decoder_type="Decoder",
            **kwargs,
        )
    elif model_name == 'DINOv3_Adapter_FRM':
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            num_modalities=num_modalities,
            adapter_type="SampleAdapter",
            decoder_type="Decoder_FRM",
            **kwargs,
        )
    elif model_name == 'DINOv3_Adapter_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_FRM_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_FRM_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_PRN_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_PRN_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_FRM",
                                  **kwargs)
    elif model_name == 'DINOv3_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_FRM_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_PRN_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_PRN_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_Baseline':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="LinearHead",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="LinearHead",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM_MMFF_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_FRM_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_FRM_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_FRM_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_ResNet50':
        model = ResNetSegmentModule(n_classes=n_classes,
                                    use_lora=use_lora,
                                    r=r,
                                    num_modalities=num_modalities,
                                    decoder_type="Decoder")

    return model
