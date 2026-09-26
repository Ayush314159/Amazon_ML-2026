import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from er.candidates import _fuse_rows  # noqa: E402


class TestLateFusion(unittest.TestCase):
    def test_keeps_best_score_per_row_and_ranks(self):
        a_rows, a_sc = np.array([5, 7, 9], dtype=np.int32), np.array([0.9, 0.4, 0.2], dtype=np.float32)
        b_rows, b_sc = np.array([7, 3], dtype=np.int32), np.array([0.8, 0.5], dtype=np.float32)
        rows, sc = _fuse_rows([a_rows, b_rows], [a_sc, b_sc], k=10)
        self.assertEqual(rows.tolist(), [5, 7, 3, 9])          # 7 takes its better score 0.8
        np.testing.assert_allclose(sc, [0.9, 0.8, 0.5, 0.2])

    def test_top_k_and_empty(self):
        rows, sc = _fuse_rows([np.array([1, 2, 3], dtype=np.int32)], [np.array([0.1, 0.3, 0.2], dtype=np.float32)], 2)
        self.assertEqual(rows.tolist(), [2, 3])
        rows, sc = _fuse_rows([np.zeros(0, dtype=np.int32)], [np.zeros(0, dtype=np.float32)], 5)
        self.assertEqual(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
