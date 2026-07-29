"""Pinned NAF inference module used by the optional P2 residual experiment.

This file is adapted from valeoai/NAF commit
37f2dfc180f2de53d98bd601109c0da0dd6b0f43 (Apache-2.0).  The rotary
position implementation is reused from the DINOv3 copy already distributed
with this repository.  See ``third_party/NAF_NOTICE.md`` for provenance.

Only the released NAF architecture needed for inference is included here.  A
recent NATTEN build exposing ``natten.na2d`` is required when this module is
imported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.layers.attention import rope_apply
from dinov3.layers.rope_position_encoding import RopePositionEmbedding

try:
    from natten import na2d
except ImportError as exc:  # pragma: no cover - exercised on the GPU server
    raise ImportError(
        "NAF P2 requires a recent NATTEN installation exposing natten.na2d"
    ) from exc


class SpatialRoPE(RopePositionEmbedding):
    """Apply DINOv3 axial RoPE directly to a BCHW feature map."""

    def __init__(self, embed_dim: int, *, num_heads: int, **kwargs) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if channels % self.num_heads != 0:
            raise ValueError(
                f"RoPE channels ({channels}) must be divisible by heads "
                f"({self.num_heads})"
            )

        head_dim = channels // self.num_heads
        x_heads = x.reshape(batch, self.num_heads, head_dim, height * width)
        x_heads = x_heads.permute(0, 1, 3, 2)
        sin, cos = super().forward(H=height, W=width)
        x_heads = rope_apply(x_heads, sin, cos)
        return (
            x_heads.permute(0, 1, 3, 2)
            .reshape(batch, channels, height, width)
        )


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )
        self.activation = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.conv2(self.activation(self.norm2(x)))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        # The released NAF encoder uses residual=False.
        return x


def make_encoder(
    in_channels: int,
    hidden_channels: int,
    *,
    kernel_size: int,
    residual_kernel_size: int,
    num_layers: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
        ),
        *[
            EncoderBlock(
                hidden_channels,
                hidden_channels,
                kernel_size=residual_kernel_size,
            )
            for _ in range(num_layers)
        ],
    )


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        heads_rope: int = 1,
        use_encoder: bool = True,
        rope_base: float | None = None,
        rope_rescale: float | None = None,
        img_layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels
        self.encoder = make_encoder(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            residual_kernel_size=1,
            num_layers=img_layers,
        )
        self.sem_encoder = make_encoder(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            residual_kernel_size=3,
            num_layers=img_layers,
        )
        self.rope = SpatialRoPE(
            embed_dim=out_channels,
            num_heads=heads_rope,
            base=rope_base,
            rescale_coords=rope_rescale,
        )

    def forward_encoder(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        if self.use_encoder:
            x = torch.cat([self.encoder(x), self.sem_encoder(x)], dim=1)
        return F.adaptive_avg_pool2d(x, output_size=output_size)

    def forward(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        out_height, out_width = output_size
        if x.shape[-2] > 4 * out_height or x.shape[-1] > 4 * out_width:
            x = F.interpolate(
                x,
                size=(
                    min(x.shape[-2], 4 * out_height, 4 * out_width),
                    min(x.shape[-1], 4 * out_width, 4 * out_height),
                ),
                mode="bilinear",
                align_corners=False,
            )
        return self.rope(self.forward_encoder(x, output_size))


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: tuple[int, int] = (9, 9),
        backend: str = "cutlass-fna",
        q_tile_shape: tuple[int, int] | None = None,
        kv_tile_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("NAF dim must be divisible by its attention heads")
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.backend = backend
        self.q_tile_shape = q_tile_shape
        self.kv_tile_shape = kv_tile_shape

    def _resize(self, x: Tensor, size: tuple[int, int], dtype: torch.dtype) -> Tensor:
        x = F.interpolate(x, size=size, mode="nearest-exact")
        batch, channels, height, width = x.shape
        head_dim = channels // self.num_heads
        return (
            x.reshape(batch, self.num_heads, head_dim, height, width)
            .permute(0, 3, 4, 1, 2)
            .to(dtype)
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        query_size = q.shape[-2:]
        key_size = k.shape[-2:]
        if any(q_dim % k_dim for q_dim, k_dim in zip(query_size, key_size)):
            raise ValueError(
                f"NAF target size {query_size} must be divisible by "
                f"feature size {key_size}"
            )
        dilation = tuple(q_dim // k_dim for q_dim, k_dim in zip(query_size, key_size))

        batch, channels, height, width = q.shape
        head_dim = channels // self.num_heads
        q = q.reshape(batch, self.num_heads, head_dim, height, width).permute(
            0, 3, 4, 1, 2
        )
        k = self._resize(k, query_size, q.dtype)
        v = self._resize(v, query_size, q.dtype)
        out = na2d(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            dilation=dilation,
            stride=1,
            backend=self.backend,
            q_tile_shape=self.q_tile_shape,
            kv_tile_shape=self.kv_tile_shape,
        )
        return out.permute(0, 3, 4, 1, 2).reshape(
            batch, channels, height, width
        )


class NAF(nn.Module):
    """Released zero-shot Neighborhood Attention Filtering architecture."""

    def __init__(
        self,
        dim: int = 256,
        heads_attn: int = 4,
        heads_rope: int = 4,
        kernel_size: int = 9,
        use_encoder: bool = True,
        rope_base: float = 100.0,
        rope_rescale: float = 2.0,
        img_layers: int = 2,
        backend: str = "cutlass-fna",
        q_tile_shape: tuple[int, int] | None = None,
        kv_tile_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            in_channels=3,
            out_channels=dim,
            heads_rope=heads_rope,
            use_encoder=use_encoder,
            rope_base=rope_base,
            rope_rescale=rope_rescale,
            img_layers=img_layers,
        )
        self.upsampler = CrossAttention(
            dim=dim,
            num_heads=heads_attn,
            kernel_size=(kernel_size, kernel_size),
            backend=backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
        )

    def forward(
        self, image: Tensor, features: Tensor, output_size: tuple[int, int]
    ) -> Tensor:
        queries = self.image_encoder(image, output_size=output_size)
        keys = F.adaptive_avg_pool2d(queries, output_size=features.shape[-2:])
        return self.upsampler(queries, keys, features)


def load_released_naf(
    checkpoint_path: str,
    *,
    backend: str = "cutlass-fna",
    q_tile_shape: tuple[int, int] | None = None,
    kv_tile_shape: tuple[int, int] | None = None,
) -> NAF:
    """Strictly load the released standalone NAF state dict and freeze it."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("NAF checkpoint must contain a state-dict mapping")

    model = NAF(
        backend=backend,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
    )
    model.load_state_dict(checkpoint, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model
