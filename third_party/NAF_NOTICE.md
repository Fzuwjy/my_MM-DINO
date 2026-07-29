# NAF third-party notice

The optional P2 residual experiment contains an adapted, inference-only copy
of the NAF architecture from:

- Project: **NAF: Zero-Shot Feature Upsampling via Neighborhood Attention
  Filtering**
- Upstream: <https://github.com/valeoai/NAF>
- Pinned revision: `37f2dfc180f2de53d98bd601109c0da0dd6b0f43`
- Upstream files adapted: `src/model/naf.py`,
  `src/layers/attentions.py`, and `src/layers/convolutions.py`
- License: Apache License 2.0; a copy is provided in
  [`NAF_LICENSE`](NAF_LICENSE)

The NAF implementation's rotary-position operation was sourced from DINOv3.
This repository already distributes that implementation under the DINOv3
license in `dinov3/layers/rope_position_encoding.py` and
`dinov3/layers/attention.py`; the experiment imports those files instead of
duplicating them.

Local changes are limited to packaging the released inference graph as a
lazy optional dependency, removing unused training/evaluation interfaces,
using relative project imports, adding shape validation, and adding a strict
checkpoint loader. The released parameter names and tensor operations used by
the default NAF checkpoint are retained.
