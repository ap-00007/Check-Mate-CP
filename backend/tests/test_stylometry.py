"""
Unit tests for NLP Stylometry, 5-module linguistic fingerprints, and comparisons.
"""
import pytest
from app.ml.stylometry import (
    ALL_NLP_FEATURE_COLS,
    FUNCTION_WORDS,
    moving_average_ttr,
    extract_stylometric_features,
    compute_nlp_module_similarities,
    build_nlp_feature_comparison_rows,
)

SAMPLE_TEXT_A = (
    "The historical development of handwritten documents reveals deeply ingrained "
    "patterns of personal expression. Although modern communication has largely transitioned "
    "to digital keystrokes, handwriting remains an intimate cognitive signature. Each writer "
    "unconsciously modulates letter proportions, stroke velocities, and grammatical structures. "
    "When analyzing authorship, stylometric markers provide robust evidence across varying topics."
)

SAMPLE_TEXT_B = (
    "Although digital systems dominate modern correspondence, handwritten manuscripts preserve "
    "invaluable linguistic fingerprints. When researchers study personal expression, they observe "
    "consistent lexical richness and syntactic pacing. Each author exhibits distinct subconscious "
    "habits in function word placement and clause organization, providing durable verification signals."
)

SHORT_TEXT = "Just a short test note."


def test_all_116_features_extracted():
    feats = extract_stylometric_features(SAMPLE_TEXT_A)
    assert len(feats) >= 116
    for col in ALL_NLP_FEATURE_COLS:
        assert col in feats, f"Missing feature: {col}"
        assert isinstance(feats[col], (int, float))


def test_lexical_diversity_metrics():
    words = SAMPLE_TEXT_A.lower().split()
    mattr = moving_average_ttr(words, window=20)
    assert 0.0 <= mattr <= 1.0

    feats = extract_stylometric_features(SAMPLE_TEXT_A)
    assert 0.0 <= feats["ttr"] <= 1.0
    assert 0.0 <= feats["hapax_ratio"] <= 1.0
    assert feats["avg_word_length"] > 2.0


def test_syntactic_and_structural_features():
    feats = extract_stylometric_features(SAMPLE_TEXT_A)
    assert feats["noun_rate"] >= 0.0
    assert feats["verb_rate"] >= 0.0
    assert feats["noun_to_verb_ratio"] > 0.0
    assert feats["avg_sentence_length"] > 5.0
    assert feats["dependency_depth_mean"] >= 1.0


def test_function_words():
    feats = extract_stylometric_features(SAMPLE_TEXT_A)
    assert feats["function_word_rate"] > 0.0
    for w in FUNCTION_WORDS[:10]:
        assert f"fw_{w}" in feats


def test_module_similarities():
    fa = extract_stylometric_features(SAMPLE_TEXT_A)
    fb = extract_stylometric_features(SAMPLE_TEXT_B)
    sims = compute_nlp_module_similarities(fa, fb, char_ngram_cosine=0.85)

    expected_modules = {"lexical", "syntactic", "structural", "punctuation", "character_nlp"}
    assert set(sims.keys()) == expected_modules
    for k, v in sims.items():
        assert 0.0 <= v <= 1.0


def test_feature_comparison_rows():
    fa = extract_stylometric_features(SAMPLE_TEXT_A)
    fb = extract_stylometric_features(SAMPLE_TEXT_B)
    rows = build_nlp_feature_comparison_rows(fa, fb, word_count_a=60, word_count_b=55)

    assert len(rows) > 0
    first = rows[0]
    assert "feature" in first
    assert "reference" in first
    assert "comparison" in first
    assert "absolute_difference" in first
    assert "relative_difference" in first
    assert "interpretation" in first
    assert "reliability" in first
    assert "High reliability" in first["reliability"]
