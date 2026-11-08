"""Unit tests for KYC Layer 3 fuzzy name matcher."""
import pytest
from backend.app.kyc_matcher import (
    MatchResult,
    match_name,
    normalize_name,
    token_set_similarity,
)
from backend.app.registries import (
    GSTINRecord,
    UdyamRecord,
    load_gstin_registry,
    load_udyam_registry,
)


class TestNormalizeName:
    def test_lowercase_conversion(self):
        assert normalize_name("BALAJI DENTAL CARE") == "balaji dental care"

    def test_strip_common_suffixes_and_noise_words(self):
        # All 11 keywords: pvt, ltd, llp, private, limited, partnership, proprietorship, m/s, messrs, shri, smt
        assert normalize_name("M/s Balaji Dental Care Pvt Ltd") == "balaji dental care"
        assert normalize_name("Balaji Dental Care Private Limited") == "balaji dental care"
        assert normalize_name("Messrs Ramesh & Sons LLP") == "ramesh sons"
        assert normalize_name("Shri Ramesh Kumar Partnership") == "ramesh kumar"
        assert normalize_name("Smt Priya Sharma Proprietorship") == "priya sharma"

    def test_strip_punctuation_but_keep_spaces(self):
        assert normalize_name("Balaji, Dental - Care (Clinic)!") == "balaji dental care clinic"
        assert normalize_name("M/s. Balaji Dental Care Pvt. Ltd.") == "balaji dental care"

    def test_collapse_multiple_spaces(self):
        assert normalize_name("   Balaji    Dental   Care   ") == "balaji dental care"

    def test_none_and_empty_inputs(self):
        assert normalize_name(None) == ""
        assert normalize_name("") == ""
        assert normalize_name("    ") == ""
        assert normalize_name(123) == ""

    def test_string_composed_only_of_suffixes(self):
        assert normalize_name("Pvt Ltd LLP") == ""
        assert normalize_name("m/s") == ""
        assert normalize_name("Private Limited") == ""

    def test_suffixes_as_substrings_not_stripped(self):
        # "shri" inside "shrikanth" should not be stripped
        assert normalize_name("Shrikanth Enterprises") == "shrikanth enterprises"

    def test_alphanumeric_names(self):
        assert normalize_name("Studio 77 Media") == "studio 77 media"


class TestTokenSetSimilarity:
    def test_exact_match(self):
        assert token_set_similarity("Balaji Dental Care", "Balaji Dental Care") == 1.0

    def test_exact_match_after_normalization(self):
        score = token_set_similarity("M/s Balaji Dental Care Pvt. Ltd.", "Balaji Dental Care")
        assert score == 1.0

    def test_word_order_independence(self):
        # Jaccard handles token permutations
        score = token_set_similarity("Care Dental Balaji", "Balaji Dental Care")
        assert score == 1.0

    def test_ocr_typo_rescued_by_sequence_matcher(self):
        # Single-letter OCR misread: 'Dentel' instead of 'Dental'
        score = token_set_similarity("Balaji Dentel Care", "Balaji Dental Care")
        assert score >= 0.8  # SequenceMatcher ratio should easily exceed 0.8

    def test_completely_different_names(self):
        score = token_set_similarity("Saffron Bakers", "Zenith Diagnostics")
        assert score < 0.6

    def test_none_and_empty_graceful(self):
        assert token_set_similarity("", "Balaji Dental Care") == 0.0
        assert token_set_similarity("Balaji Dental Care", "") == 0.0
        assert token_set_similarity(None, "Balaji Dental Care") == 0.0
        assert token_set_similarity(None, None) == 0.0
        assert token_set_similarity("Pvt Ltd", "Private Limited") == 0.0


class TestMatchName:
    @pytest.fixture
    def mock_gstin_rec(self):
        return GSTINRecord(
            gstin="03PJNXS4821G1ZB",
            legal_name="Sneha Rao (Prop. Saffron Bakers)",
            trade_name="Saffron Bakers",
            constitution="Proprietorship",
            principal_place_of_business="Shop/Unit 83, Market Yard, Ludhiana, Punjab - 416059",
            state="Punjab",
            registration_date="2019-04-15",
            status="Active",
        )

    @pytest.fixture
    def mock_udyam_rec(self):
        return UdyamRecord(
            udyam_number="UDYAM-KL-00-3078801",
            enterprise_name="Balaji Dental Care LLP",
            enterprise_type="Micro",
            major_activity="Services",
            nic_description="Dental clinic",
            state="Kerala",
            city="Kochi",
            date_of_registration="2023-12-25",
        )

    def test_gstin_matches_trade_name(self, mock_gstin_rec):
        res = match_name("M/s Saffron Bakers", mock_gstin_rec)
        assert isinstance(res, MatchResult)
        assert res.registry_name == "Saffron Bakers"
        assert res.score == 1.0
        assert res.verdict == "match"

    def test_gstin_matches_legal_name_when_trade_name_does_not(self, mock_gstin_rec):
        # OCR extracted the proprietor name
        res = match_name("Sneha Rao", mock_gstin_rec)
        assert res.registry_name == "Sneha Rao (Prop. Saffron Bakers)"
        assert res.score > 0.0

    def test_udyam_matches_enterprise_name(self, mock_udyam_rec):
        res = match_name("Balaji Dental Care", mock_udyam_rec)
        assert res.registry_name == "Balaji Dental Care LLP"
        assert res.score == 1.0
        assert res.verdict == "match"

    def test_verdict_thresholds(self, mock_udyam_rec):
        # >= 0.8: match
        res_match = match_name("Balaji Dental Care", mock_udyam_rec)
        assert res_match.verdict == "match"
        assert res_match.score >= 0.8

        # Mismatch (< 0.6)
        res_mismatch = match_name("Completely Unrelated Store", mock_udyam_rec)
        assert res_mismatch.verdict == "mismatch"
        assert res_mismatch.score < 0.6

    def test_review_verdict_range(self):
        rec = UdyamRecord(
            udyam_number="UDYAM-KL-00-1111111",
            enterprise_name="Balaji Dental Clinic",
            enterprise_type="Micro",
            major_activity="Services",
            nic_description="Clinic",
            state="Kerala",
            city="Kochi",
            date_of_registration="2023-01-01",
        )
        # "Balaji Dental Care" vs "Balaji Dental Clinic"
        res = match_name("Balaji Dental Care", rec)
        assert res.verdict == "review"
        assert 0.6 <= res.score < 0.8

    def test_none_and_empty_graceful_handling(self, mock_gstin_rec, mock_udyam_rec):
        # None ocr_name
        res1 = match_name(None, mock_gstin_rec)
        assert res1.score == 0.0
        assert res1.verdict == "mismatch"
        assert res1.ocr_name == ""

        # Empty string ocr_name
        res2 = match_name("   ", mock_udyam_rec)
        assert res2.score == 0.0
        assert res2.verdict == "mismatch"
        assert res2.ocr_name == ""

        # None registry_record
        res3 = match_name("Balaji Dental Care", None)
        assert res3.score == 0.0
        assert res3.verdict == "mismatch"
        assert res3.registry_name == ""

    def test_real_registry_records(self):
        gst_reg = load_gstin_registry()
        udyam_reg = load_udyam_registry()

        gst_rec = gst_reg["32TTNXB7384T1ZW"]
        res_gst = match_name("M/s Balaji Dental Care", gst_rec)
        assert res_gst.verdict == "match"
        assert res_gst.score == 1.0

        udyam_rec = udyam_reg["UDYAM-KL-00-3078801"]
        res_udy = match_name("Balaji Dental Care Pvt Ltd", udyam_rec)
        assert res_udy.verdict == "match"
        assert res_udy.score == 1.0
