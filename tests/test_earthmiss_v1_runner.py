"""Small policy checks for the EarthMiss V1 launcher."""

import random
import unittest

from scripts.train_earthmiss_missing_v1 import train_state, validation_states


class EarthMissV1RunnerTest(unittest.TestCase):
    def test_run_policies_and_validation_targets(self):
        rng = random.Random(42)
        self.assertEqual(train_state("A", rng), "sar")
        self.assertEqual(train_state("B", rng), "full")
        self.assertEqual(validation_states("A"), ("sar",))
        self.assertEqual(validation_states("B"), ("sar", "full"))
        self.assertEqual(validation_states("C"), ("sar", "full"))

    def test_run_c_uses_only_full_and_sar_and_is_deterministic(self):
        first_rng = random.Random(99)
        second_rng = random.Random(99)
        first = [train_state("C", first_rng) for _ in range(20)]
        second = [train_state("C", second_rng) for _ in range(20)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"full", "sar"})


if __name__ == "__main__":
    unittest.main()
