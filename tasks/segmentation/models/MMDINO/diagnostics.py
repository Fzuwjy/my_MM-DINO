"""Diagnostic-only model adapters for missing-modality audits."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureZeroSampleAdapter(nn.Module):
    """Zero one processed modality while preserving released fusion weights.

    This wrapper is only for the legacy zero/no-renormalization endpoint.  It
    keeps the normal two-input model path: both backbones execute, both learned
    modality weights remain in the denominator, and the fused pyramid is still
    emitted in both decoder slots.  The selected modality is zeroed after the
    delegate's projection and resize layers and before the weighted sum.
    """

    def __init__(
        self,
        delegate: nn.Module,
        *,
        zero_modality_index: int | None,
    ) -> None:
        super().__init__()
        if int(getattr(delegate, "num_modalities", 0)) != 2:
            raise ValueError("Feature-zero diagnostics require a two-modality adapter")
        if zero_modality_index not in (None, 0, 1):
            raise ValueError("zero_modality_index must be None, 0 (RGB), or 1 (SAR)")
        if bool(getattr(delegate, "use_naf", False)):
            raise ValueError("Feature-zero diagnostics are not defined for NAF adapters")

        self.delegate = delegate
        self.zero_modality_index = zero_modality_index
        self.num_modalities = 2

    def forward(
        self,
        *features_list,
        patch_h=None,
        patch_w=None,
        guidance=None,
        modality_indices=None,
    ):
        if len(features_list) != 2:
            raise ValueError(
                "Feature-zero diagnostics preserve the full two-input path; "
                f"got {len(features_list)} feature sets"
            )
        if patch_h is None or patch_w is None:
            raise ValueError("patch_h and patch_w are required")

        if modality_indices is not None:
            modality_indices = tuple(modality_indices)
            if modality_indices != (0, 1):
                raise ValueError(
                    "Feature-zero diagnostics require the canonical (RGB, SAR) slots"
                )

        if self.zero_modality_index is None:
            return self.delegate(
                *features_list,
                patch_h=patch_h,
                patch_w=patch_w,
                guidance=guidance,
                modality_indices=modality_indices,
            )

        outputs = [[], []]
        for layer_index, modality_features in enumerate(zip(*features_list)):
            processed_features = []
            for feature in modality_features:
                feature = feature.permute(0, 2, 1).reshape(
                    feature.shape[0], feature.shape[-1], patch_h, patch_w
                )
                feature = self.delegate.projects[layer_index](feature)
                feature = self.delegate.resize_layers[layer_index](feature)
                processed_features.append(feature)

            processed_features[self.zero_modality_index] = torch.zeros_like(
                processed_features[self.zero_modality_index]
            )

            # Recompute per slot exactly as the released SampleAdapter does.
            # The denominator intentionally includes the zeroed modality.
            for output_slot in range(2):
                weights = [
                    torch.sigmoid(
                        self.delegate.modality_weights[f"weight_modality_{index}"]
                    )
                    for index in range(2)
                ]
                total_weight = sum(weights)
                normalized_weights = [weight / total_weight for weight in weights]
                fused_feature = sum(
                    weight * feature
                    for weight, feature in zip(
                        normalized_weights, processed_features, strict=True
                    )
                )
                outputs[output_slot].append(fused_feature)

        return outputs
