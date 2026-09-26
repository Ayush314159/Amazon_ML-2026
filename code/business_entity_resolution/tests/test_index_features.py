import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from er.index import record_features  # noqa: E402


class TestPhoneticFeaturePollution(unittest.TestCase):
    def test_legal_forms_excluded_from_phonetic_namespace(self):
        # "Apex Systems Pvt Ltd" should not contribute p:pft (pvt) or p:lt (ltd) --
        # those collapse ~10-28% of all records (Phase 1 legal-suffix stats) into
        # the same phonetic bucket and drown out the genuinely distinguishing tokens.
        feats = record_features("apex systems pvt ltd", "")
        p_feats = {f for f in feats if f.startswith("p:")}
        self.assertNotIn("p:pft", p_feats)
        self.assertNotIn("p:lt", p_feats)
        self.assertEqual(p_feats, {"p:pks", "p:stms"})

    def test_content_words_still_produce_phonetic_keys(self):
        feats = record_features("systems", "")
        self.assertIn("p:stms", feats)


if __name__ == "__main__":
    unittest.main()
