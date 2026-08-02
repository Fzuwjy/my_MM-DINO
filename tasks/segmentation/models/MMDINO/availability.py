"""Canonical two-slot modality availability for EarthMiss experiments."""

import torch


AVAILABILITY_STATES = {
    "full": (True, True),
    "sar": (False, True),
    "rgb": (True, False),
}


def canonical_availability(state, *, batch_size, device=None):
    try:
        values = AVAILABILITY_STATES[state]
    except KeyError as error:
        choices = ", ".join(AVAILABILITY_STATES)
        raise ValueError(f"Unknown availability state {state!r}; choose from {choices}") from error
    return torch.tensor(values, dtype=torch.bool, device=device).repeat(batch_size, 1)


def active_modality_indices(availability, *, batch_size, num_modalities):
    """Validate availability and return active slots for a homogeneous batch."""
    if availability is None:
        return tuple(range(num_modalities))

    availability = torch.as_tensor(availability)
    if availability.ndim == 1:
        availability = availability.unsqueeze(0)
    if availability.ndim != 2 or availability.shape[1] != num_modalities:
        raise ValueError(
            f"availability must have shape [B, {num_modalities}], got {tuple(availability.shape)}"
        )
    if availability.shape[0] == 1 and batch_size > 1:
        availability = availability.expand(batch_size, -1)
    if availability.shape[0] != batch_size:
        raise ValueError(
            f"availability batch size {availability.shape[0]} does not match inputs {batch_size}"
        )

    availability = availability.to(dtype=torch.bool)
    if not torch.equal(availability, availability[:1].expand_as(availability)):
        raise ValueError("V1 requires one availability state per batch")
    if not availability[0].any():
        raise ValueError("At least one modality must be available")
    return tuple(torch.nonzero(availability[0], as_tuple=False).flatten().tolist())
