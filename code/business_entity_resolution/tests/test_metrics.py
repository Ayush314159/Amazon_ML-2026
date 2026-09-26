import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from er.data import validate_submission, write_id_lists  # noqa: E402
from er.metrics import blocking_metrics, entity_scores, f_beta, macro_f05  # noqa: E402


class TestEntityF05(unittest.TestCase):
    def test_pdf_worked_example(self):
        # amazon_ml.pdf: pred [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
        p, r, f = entity_scores({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
        self.assertAlmostEqual(p, 2 / 3)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 0.714, places=3)

    def test_singleton_rules(self):
        self.assertEqual(entity_scores(set(), set())[2], 1.0)
        self.assertEqual(entity_scores({"S2-1"}, set())[2], 0.0)

    def test_empty_prediction_for_non_singleton(self):
        self.assertEqual(entity_scores(set(), {"S2-1"})[2], 0.0)

    def test_no_overlap(self):
        self.assertEqual(entity_scores({"S2-9"}, {"S2-1"})[2], 0.0)

    def test_precision_weighted_more_than_recall(self):
        high_p = f_beta(1.0, 0.5)
        high_r = f_beta(0.5, 1.0)
        self.assertGreater(high_p, high_r)

    def test_macro_average_and_missing_predictions(self):
        truth = {"S1-a": frozenset({"S2-1", "S3-2"}), "S1-b": frozenset(), "S1-c": frozenset({"S2-5"})}
        pred = {"S1-a": {"S2-1", "S3-2"}, "S1-b": set()}  # S1-c missing -> empty prediction
        res = macro_f05(pred, truth)
        self.assertAlmostEqual(res["f05"], (1.0 + 1.0 + 0.0) / 3)
        self.assertAlmostEqual(res["f05_singletons"], 1.0)
        self.assertAlmostEqual(res["f05_non_singletons"], 0.5)


class TestBlockingMetrics(unittest.TestCase):
    def test_recall_and_oracle(self):
        truth = {"q1": frozenset({"S2-1", "S3-1"}), "q2": frozenset(), "q3": frozenset({"S2-9"})}
        cands = {"q1": {"S2-1", "S2-7"}, "q2": {"S2-3"}, "q3": set()}
        m = blocking_metrics(cands, truth, ["q1", "q2", "q3"], pool_size=100)
        self.assertAlmostEqual(m["pair_recall"], 1 / 3)
        self.assertAlmostEqual(m["entity_full_coverage"], 0.0)
        # q1 oracle recall 0.5 -> f(1, .5); q2 singleton -> 1; q3 no hit -> 0
        self.assertAlmostEqual(m["oracle_f05"], (f_beta(1.0, 0.5) + 1.0 + 0.0) / 3)
        self.assertEqual(m["cands_total"], 3)
        self.assertAlmostEqual(m["reduction_ratio"], 1 - 3 / 300)
        self.assertTrue(math.isclose(m["zero_cand_share"], 1 / 3))


class TestSubmissionValidation(unittest.TestCase):
    def test_valid_and_invalid_files(self):
        s1 = ["S1-1", "S1-2", "S1-3"]
        targets = {"S2-1", "S2-2", "S3-1"}
        with tempfile.TemporaryDirectory() as d:
            mp, cp = os.path.join(d, "m.tsv"), os.path.join(d, "c.tsv")
            write_id_lists(cp, {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"]}, s1, "candidate_entity_ids")
            write_id_lists(mp, {"S1-1": ["S2-1"]}, s1, "matched_entity_ids")
            self.assertEqual(validate_submission(mp, cp, s1, targets), [])
            # match outside candidates + unknown id
            write_id_lists(mp, {"S1-1": ["S2-2"], "S1-3": ["S2-404"]}, s1, "matched_entity_ids")
            issues = validate_submission(mp, cp, s1, targets)
            self.assertTrue(any("not a subset" in i for i in issues))
            self.assertTrue(any("not Source-2/3 ids" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
