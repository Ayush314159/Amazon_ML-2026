import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from er.normalize import (addr_set_key, name_core_key, normalize_addr,  # noqa: E402
                          normalize_name, phonetic_key)


class TestNormalize(unittest.TestCase):
    def test_case_accents_punctuation(self):
        self.assertEqual(normalize_name("Prairie Grand ópportunity"), "prairie grand opportunity")
        self.assertEqual(normalize_name("The Éar Nose & Throat"), "the ear nose and throat")
        self.assertEqual(normalize_name("Orelee's Barbershop"), "orelees barbershop")

    def test_encoding_corruption_dropped(self):
        self.assertEqual(normalize_name("ARDATH MARTIN FR�EDOM INC"), "ardath martin fredom inc")

    def test_ocr_digit_fix_but_not_ordinals_or_numbers(self):
        self.assertEqual(normalize_name("t0rrescoffee.com"), "torrescoffee com")
        self.assertEqual(normalize_name("Prairie 6rand"), "prairie grand")
        self.assertEqual(normalize_name("21st Century 101 Crypto"), "21st century 101 crypto")

    def test_legal_forms_canonical_and_core_key(self):
        a = name_core_key(normalize_name("Sorrells & Hanson LLC"))
        b = name_core_key(normalize_name("LLC SORRELLS & HANSON"))
        self.assertEqual(a, b)
        self.assertEqual(normalize_name("UW Runners Private Limited"), "uw runners pvt ltd")
        self.assertEqual(name_core_key(normalize_name("Lille Anciens SARL")), "anciens lille")

    def test_address_canonicalization_and_order(self):
        self.assertEqual(normalize_addr("00132 Lemonwood Lane, Hollister"), "132 lemonwood ln hollister")
        a = addr_set_key(normalize_addr("OH, Columbus, 5559 Orville Avenue"))
        b = addr_set_key(normalize_addr("5559 Orville Ave, Columbus, OH"))
        self.assertEqual(a, b)

    def test_indic_transliteration(self):
        self.assertEqual(normalize_name("राम मार्केटिंग"), "ram marketing")
        self.assertEqual(normalize_name("बालाजी डेवलपर्स"), "balaji devalapars")
        # same table serves other Brahmic blocks (Kannada, Bengali)
        self.assertEqual(normalize_name("ಸಿಸ್ಟಮ್ಸ್"), "sistams")
        self.assertEqual(normalize_name("প্রজেক্টস"), "prajektas")

    def test_phonetic_key_bridges_transliteration(self):
        for latin, native in [("systems", "सिस्टम्स"), ("apex", "एपेक्स"), ("management", "मैनेजमेंट"),
                              ("developers", "डेवलपर्स"), ("construction", "कंस्ट्रक्शन")]:
            self.assertEqual(phonetic_key(latin), phonetic_key(normalize_name(native)), (latin, native))
        self.assertEqual(phonetic_key("1604"), "")
        self.assertEqual(phonetic_key("a"), "")

    def test_empty(self):
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_addr(""), "")


if __name__ == "__main__":
    unittest.main()
