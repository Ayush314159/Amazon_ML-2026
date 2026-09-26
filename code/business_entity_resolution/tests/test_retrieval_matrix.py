import os
import sys
import types
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from er.blocking import _retrieval_matrix  # noqa: E402


def _q(cols):
    """One-row query matrix over the given feature columns (weight 1)."""
    return sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (np.zeros(len(cols), dtype=int), cols)), shape=(1, 10))


class TestRetrievalMatrix(unittest.TestCase):
    def setUp(self):
        # feature df: 0 -> 10 (rare), 1 -> 30k, 2 -> 60k, 3 -> 150k, 4 -> 500k
        self.index = types.SimpleNamespace(df=np.array([10, 30_000, 60_000, 150_000, 500_000, 0, 0, 0, 0, 0]))

    def test_default_keeps_only_below_cap(self):
        R = _retrieval_matrix(self.index, _q([0, 1, 2]), cap_df=20_000)
        self.assertEqual(sorted(R.indices.tolist()), [0])

    def test_single_rarest_fallback(self):
        R = _retrieval_matrix(self.index, _q([2, 3, 4]), cap_df=20_000)
        self.assertEqual(R.indices.tolist(), [2])

    def test_top_up_to_min_feats_within_hard_cap(self):
        R = _retrieval_matrix(self.index, _q([0, 1, 2, 3, 4]), cap_df=20_000, min_feats=3, hard_cap_df=200_000)
        self.assertEqual(sorted(R.indices.tolist()), [0, 1, 2])
        # the hard cap stops the top-up: only features with df <= 50k may be added
        R = _retrieval_matrix(self.index, _q([0, 1, 2, 3]), cap_df=20_000, min_feats=3, hard_cap_df=50_000)
        self.assertEqual(sorted(R.indices.tolist()), [0, 1])


if __name__ == "__main__":
    unittest.main()
