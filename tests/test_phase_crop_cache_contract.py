"""Pure-CPU producer/replay contract tests for the phase-crop cache."""

from __future__ import annotations

import copy
import hashlib
import struct
import unittest

import numpy as np

from scripts.cache_whu_phase_crop_logits import (
    ARTIFACT_TYPE as PRODUCER_ARTIFACT_TYPE,
    PHASE_ORDER as PRODUCER_PHASE_ORDER,
    SCHEMA_VERSION as PRODUCER_SCHEMA_VERSION,
    _crop_ids_sha256 as producer_crop_ids_sha256,
    _hash_descriptor as producer_hash_descriptor,
    _protocol as producer_protocol,
)
from scripts.evaluate_whu_phase_cached_replay import (
    CACHE_ARTIFACT_TYPE as REPLAY_CACHE_ARTIFACT_TYPE,
    CACHE_SCHEMA_VERSION as REPLAY_CACHE_SCHEMA_VERSION,
    PHASE_ORDER as REPLAY_PHASE_ORDER,
    _crop_id_sha256 as replay_crop_ids_sha256,
    _validate_hash_descriptor as replay_validate_hash_descriptor,
)


class PhaseCropCacheContractTest(unittest.TestCase):
    def test_schema_and_short_protocol_encodings_are_exact(self) -> None:
        protocol = producer_protocol()

        self.assertEqual(PRODUCER_SCHEMA_VERSION, 1)
        self.assertEqual(PRODUCER_SCHEMA_VERSION, REPLAY_CACHE_SCHEMA_VERSION)
        self.assertEqual(
            PRODUCER_ARTIFACT_TYPE,
            "whu_phase_crop_logits_correctness_cache",
        )
        self.assertEqual(PRODUCER_ARTIFACT_TYPE, REPLAY_CACHE_ARTIFACT_TYPE)
        self.assertEqual(PRODUCER_PHASE_ORDER, REPLAY_PHASE_ORDER)
        self.assertEqual(protocol["phase_order"], list(REPLAY_PHASE_ORDER))
        self.assertEqual(protocol["raw_crop_storage_dtype"], "float32")
        self.assertEqual(protocol["dense_common_count_dtype"], "int16")
        self.assertEqual(
            protocol["crop_id_sha256_encoding"],
            "little-endian int64 bytes",
        )

    def test_crop_id_digest_is_little_endian_int64_and_shared(self) -> None:
        crop_ids = (0, 1, 256, 65_537)
        expected_digest = (
            "c5bf2e91c7a279ea32d731e864a6bb288a079ad204770063b29d51fd79492c27"
        )
        little_endian_bytes = struct.pack("<4q", *crop_ids)
        big_endian_bytes = struct.pack(">4q", *crop_ids)

        self.assertEqual(hashlib.sha256(little_endian_bytes).hexdigest(), expected_digest)
        self.assertNotEqual(hashlib.sha256(big_endian_bytes).hexdigest(), expected_digest)
        self.assertEqual(producer_crop_ids_sha256(crop_ids), expected_digest)
        self.assertEqual(replay_crop_ids_sha256(crop_ids), expected_digest)

    def test_hash_descriptor_round_trip_has_exact_fields(self) -> None:
        float_view = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        count_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        cases = (
            (
                "logits",
                float_view,
                {
                    "dtype": "float32",
                    "shape": [4, 3],
                    "nbytes": 48,
                    "array_sha256": (
                        "cfba5b61d96b3e46f0abbb32f11737b310e49438cd25dd78aad9616c7d3abdc2"
                    ),
                },
            ),
            (
                "count",
                count_array,
                {
                    "dtype": "int16",
                    "shape": [2, 3],
                    "nbytes": 12,
                    "array_sha256": (
                        "b1cd5bf03b9488553472b7264c8d53326d8d6b2aa42ab53e2d0f27387db492d5"
                    ),
                },
            ),
        )

        for name, source, expected in cases:
            with self.subTest(name=name):
                actual = np.ascontiguousarray(source)
                descriptor = producer_hash_descriptor(source)
                self.assertEqual(descriptor, expected)
                replay_validate_hash_descriptor(
                    descriptor,
                    name=name,
                    expected_dtype=expected["dtype"],
                    expected_shape=expected["shape"],
                    actual=actual,
                )

    def test_replay_rejects_tampered_hash_descriptor_fields(self) -> None:
        actual = np.arange(12, dtype=np.float32).reshape(3, 4)
        descriptor = producer_hash_descriptor(actual)
        tampered_values = {
            "dtype": "float64",
            "shape": [4, 3],
            "nbytes": descriptor["nbytes"] + 1,
            "array_sha256": "0" * 64,
        }

        for field, bad_value in tampered_values.items():
            with self.subTest(field=field):
                tampered = copy.deepcopy(descriptor)
                tampered[field] = bad_value
                with self.assertRaises((TypeError, ValueError)):
                    replay_validate_hash_descriptor(
                        tampered,
                        name="synthetic",
                        expected_dtype="float32",
                        expected_shape=(3, 4),
                        actual=actual,
                    )


if __name__ == "__main__":
    unittest.main()
