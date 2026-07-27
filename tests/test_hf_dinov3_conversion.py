from __future__ import annotations

import unittest

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from dinov3.hub.backbones import dinov3_vits16
from scripts.convert_hf_dinov3_vits16 import convert_state_dict


class HuggingFaceDINOv3ConversionTests(unittest.TestCase):
    def test_random_vits16_outputs_match_after_conversion(self):
        torch.manual_seed(7)
        hf_model = DINOv3ViTModel(
            DINOv3ViTConfig(
                num_register_tokens=4,
                layerscale_value=1e-5,
                pos_embed_rescale=2.0,
            )
        ).eval()
        reference_model = dinov3_vits16(pretrained=False).eval()
        converted = convert_state_dict(
            hf_model.state_dict(), reference_model.state_dict()
        )
        reference_model.load_state_dict(converted, strict=True)

        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            hf_tokens = hf_model(image).last_hidden_state
            features = reference_model.forward_features(image)
            reference_tokens = torch.cat(
                [
                    features["x_norm_clstoken"][:, None],
                    features["x_storage_tokens"],
                    features["x_norm_patchtokens"],
                ],
                dim=1,
            )

        torch.testing.assert_close(hf_tokens, reference_tokens, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
