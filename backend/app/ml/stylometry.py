"""
stylometry.py — Comprehensive NLP Stylometric Feature Extraction and Comparison.

Extracts a 5-module linguistic fingerprint:
1. Lexical features: TTR, MATTR (moving average), word length stats, hapax legomena,
   lexical density, stopword rate, and 69 individual function word frequencies.
2. Syntactic features: POS distributions (noun, verb, adj, adv, pron, adp, det, conj),
   noun-to-verb ratio, dependency tree depth, subordinate clause frequencies.
3. Structural features: Sentence length stats (mean, std, variance), line length,
   clause length, complex sentence ratio, and sentence opening POS patterns.
4. Punctuation patterns: Punctuation density, comma rate, period rate, question mark rate,
   exclamation mark rate, semicolon, colon, parenthesis, and quotation mark rates.
5. Character linguistic patterns: Double-letter occurrences, common morphological prefixes
   (un-, re-, in-) and suffixes (-ing, -ed, -ly, -tion).

Includes deterministic fallback parser when spaCy runtime or models are offline.
"""
from __future__ import annotations

import re
import math
from collections import Counter
from typing import Any
import numpy as np

from app.core.config import settings

FUNCTION_WORDS = [
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "of",
    "at", "by", "for", "from", "in", "into", "on", "onto", "to", "with",
    "without", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "our", "their", "this", "that", "these", "those", "not", "no", "so",
    "than", "then", "there", "here", "which", "who", "what", "when",
    "where", "why", "how",
]

NLP_MODULE_MAP = {
    "lexical": [
        "avg_word_length", "std_word_length", "ttr", "mattr_50",
        "hapax_ratio", "lexical_density", "stopword_rate", "function_word_rate",
        *[f"fw_{w}" for w in FUNCTION_WORDS],
    ],
    "syntactic": [
        "noun_rate", "verb_rate", "adj_rate", "adv_rate", "pron_rate",
        "adp_rate", "det_rate", "conj_rate", "noun_to_verb_ratio",
        "dependency_depth_mean", "subordinate_clause_rate",
    ],
    "structural": [
        "avg_sentence_length", "std_sentence_length", "var_sentence_length",
        "avg_line_length", "avg_clause_length", "complex_sentence_ratio",
        "open_pron_rate", "open_det_rate", "open_conj_rate", "open_adv_rate",
    ],
    "punctuation": [
        "punct_per_token", "comma_rate", "semicolon_rate", "colon_rate",
        "question_rate", "exclamation_rate", "parenthesis_rate",
        "period_rate", "quote_rate",
    ],
    "character_nlp": [
        "double_letter_rate", "suffix_ing_rate", "suffix_ed_rate",
        "suffix_ly_rate", "suffix_tion_rate", "prefix_un_rate",
        "prefix_re_rate", "prefix_in_rate",
    ],
}

ALL_NLP_FEATURE_COLS = [col for cols in NLP_MODULE_MAP.values() for col in cols]
NLP_META_COLS = ["meta_word_count", "meta_sentence_count", "meta_line_count"]

FEATURE_HUMAN_LABELS: dict[str, str] = {
    "avg_word_length": "Average Word Length",
    "std_word_length": "Word Length Variation",
    "ttr": "Type-Token Ratio (Vocabulary Richness)",
    "mattr_50": "Moving-Average TTR (Window 50)",
    "hapax_ratio": "Hapax Legomena Ratio (Unique Words)",
    "lexical_density": "Lexical Density (Content Words)",
    "stopword_rate": "Stopword Usage Rate",
    "function_word_rate": "Function Word Usage Rate",
    "noun_rate": "Noun Usage Rate",
    "verb_rate": "Verb Usage Rate",
    "adj_rate": "Adjective Usage Rate",
    "adv_rate": "Adverb Usage Rate",
    "pron_rate": "Pronoun Usage Rate",
    "adp_rate": "Preposition Usage Rate",
    "det_rate": "Determiner Usage Rate",
    "conj_rate": "Conjunction Usage Rate",
    "noun_to_verb_ratio": "Noun-to-Verb Ratio",
    "dependency_depth_mean": "Syntactic Tree Depth",
    "subordinate_clause_rate": "Subordinate Clause Density",
    "avg_sentence_length": "Average Sentence Length (Words)",
    "std_sentence_length": "Sentence Length Variation",
    "avg_line_length": "Average Line Length (Words)",
    "avg_clause_length": "Average Clause Length",
    "complex_sentence_ratio": "Complex Sentence Proportion",
    "punct_per_token": "Punctuation Density",
    "comma_rate": "Comma Relative Frequency",
    "period_rate": "Period Relative Frequency",
    "question_rate": "Question Mark Frequency",
    "exclamation_rate": "Exclamation Mark Frequency",
    "double_letter_rate": "Double Letter Frequency",
    "suffix_ing_rate": "'-ing' Suffix Rate",
    "suffix_ed_rate": "'-ed' Suffix Rate",
    "suffix_ly_rate": "'-ly' Suffix Rate",
    "suffix_tion_rate": "'-tion' Suffix Rate",
}

# Lazy spaCy loader
_spacy_nlp = None
_spacy_loaded = False


def get_spacy_pipeline():
    global _spacy_nlp, _spacy_loaded
    if _spacy_loaded:
        return _spacy_nlp
    _spacy_loaded = True
    try:
        import importlib
        spacy_mod = importlib.import_module("spacy")
        _spacy_nlp = spacy_mod.load(settings.spacy_model_name, disable=["ner"])
    except Exception:
        _spacy_nlp = None
    return _spacy_nlp


def moving_average_ttr(words: list[str], window: int = 50) -> float:
    """Computes moving average type-token ratio across rolling windows."""
    if len(words) < 2:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    scores = [
        len(set(words[i : i + window])) / window
        for i in range(0, len(words) - window + 1, max(1, window // 2))
    ]
    return float(np.mean(scores)) if scores else 0.0


def _fallback_parse(text: str) -> dict[str, Any]:
    """
    Deterministic rule-based NLP parser when spaCy is unavailable.
    Provides robust POS tagging, clause detection, and sentence boundaries.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Sentence boundary splitting
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]

    # Tokenization
    tokens = re.findall(r"\w+|[^\w\s]", text)
    words = [t.lower() for t in tokens if t.isalpha()]

    # Punctuation counts
    punct_tokens = [t for t in tokens if not t.isalnum() and not t.isspace()]
    punct_counts = Counter(punct_tokens)

    # Function word & stopword approximation
    common_stopwords = set(FUNCTION_WORDS) | {
        "about", "above", "after", "again", "against", "all", "am", "an", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between",
        "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
        "each", "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    }

    # Lexical POS tagging heuristics
    pos_counts: Counter[str] = Counter()
    for w in words:
        if w in {"the", "a", "an", "this", "that", "these", "those", "every", "some", "any", "my", "your", "his", "her", "our", "their"}:
            pos_counts["DET"] += 1
        elif w in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "who", "whom"}:
            pos_counts["PRON"] += 1
        elif w in {"and", "but", "or", "so", "yet", "for", "nor", "if", "because", "although", "while", "since"}:
            pos_counts["CONJ"] += 1
        elif w in {"in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "of", "off", "over", "under"}:
            pos_counts["ADP"] += 1
        elif w.endswith("ly") and len(w) > 3:
            pos_counts["ADV"] += 1
        elif w.endswith(("able", "ible", "al", "ful", "ic", "ive", "less", "ous")):
            pos_counts["ADJ"] += 1
        elif w in {"is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must"} or w.endswith(("ed", "ing", "ize", "ise")):
            pos_counts["VERB"] += 1
        else:
            pos_counts["NOUN"] += 1

    # Sentence opening POS
    open_pos = []
    for s in sentences:
        s_words = [w.lower() for w in re.findall(r"\w+", s)]
        if s_words:
            first = s_words[0]
            if first in {"i", "you", "he", "she", "it", "we", "they", "this", "that"}:
                open_pos.append("PRON")
            elif first in {"the", "a", "an"}:
                open_pos.append("DET")
            elif first in {"and", "but", "so", "if", "because", "however"}:
                open_pos.append("CONJ")
            elif first.endswith("ly") or first in {"then", "there", "now", "here"}:
                open_pos.append("ADV")
            else:
                open_pos.append("OTHER")

    # Subordinate clauses approximation via subordinating conjunctions
    sub_conj = {"if", "because", "although", "since", "unless", "while", "whereas", "as", "after", "before", "until", "when", "whenever", "where", "wherever"}
    sub_clause_count = sum(1 for w in words if w in sub_conj)

    # Complex sentences
    complex_sents = sum(
        1 for s in sentences if any(w in sub_conj for w in re.findall(r"\w+", s.lower()))
    )

    # Approximated dependency depth: scales with sentence length (log2(len) + 1)
    dep_depths = [
        float(math.log2(max(2, len(re.findall(r"\w+", s)))) + 1.2)
        for s in sentences
    ]

    return {
        "words": words,
        "sentences": sentences,
        "tokens": tokens,
        "lines": lines,
        "pos_counts": pos_counts,
        "punct_counts": punct_counts,
        "stopwords": common_stopwords,
        "open_pos": open_pos,
        "sub_clause_count": sub_clause_count,
        "complex_sents": complex_sents,
        "dep_depths": dep_depths,
    }


def extract_stylometric_features(text: str, line_count: int | None = None) -> dict[str, float]:
    """
    Extracts all 116 linguistic stylometric features matching ALL_NLP_FEATURE_COLS
    plus metadata counts. Works seamlessly with spaCy or pure Python fallback.
    """
    clean_text = text or ""
    spacy_nlp = get_spacy_pipeline()

    if spacy_nlp is not None and clean_text.strip():
        doc = spacy_nlp(clean_text)
        words = [t.text.lower() for t in doc if t.is_alpha]
        sentences = list(doc.sents)
        n_words = len(words)
        n_tokens = len(doc)
        n_sentences = len(sentences)
        lines = [ln for ln in clean_text.splitlines() if ln.strip()]
        n_lines = line_count if line_count is not None else max(1, len(lines))

        counts = Counter(words)
        pos_counts = Counter(t.pos_ for t in doc if not t.is_space)
        punct_counts = Counter(t.text for t in doc if t.is_punct)
        function_counts = Counter(w for w in words if w in FUNCTION_WORDS)

        sentence_lengths = [sum(1 for t in sent if t.is_alpha) for sent in sentences]
        word_lengths = [len(w) for w in words]
        total_punct = sum(punct_counts.values())
        sub_dep_set = {"advcl", "ccomp", "csubj", "acl", "relcl"}
        sub_clauses = sum(1 for t in doc if t.dep_ in sub_dep_set)

        complex_sents = sum(
            1 for sent in sentences if any(t.dep_ in sub_dep_set for t in sent)
        )

        open_pos = []
        for sent in sentences:
            first = next((t for t in sent if not t.is_space and not t.is_punct), None)
            if first is not None:
                open_pos.append(first.pos_)
        open_counts = Counter(open_pos)
        n_open = max(1, len(open_pos))

        noun_count = pos_counts["NOUN"] + pos_counts["PROPN"]
        verb_count = pos_counts["VERB"] + pos_counts["AUX"]

        dep_depths = [
            max((len(list(t.ancestors)) for t in sent), default=0)
            for sent in sentences
        ]
        stopword_count = sum(1 for t in doc if t.is_alpha and t.is_stop)
        lexical_density_count = sum(1 for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"})
        conj_count = pos_counts["CCONJ"] + pos_counts["SCONJ"]

    else:
        # High-fidelity fallback parser
        parsed = _fallback_parse(clean_text)
        words = parsed["words"]
        raw_sents = parsed["sentences"]
        tokens = parsed["tokens"]
        lines = parsed["lines"]
        n_lines = line_count if line_count is not None else max(1, len(lines))
        n_words = len(words)
        n_tokens = max(1, len(tokens))
        n_sentences = max(1, len(raw_sents))

        counts = Counter(words)
        pos_counts = parsed["pos_counts"]
        punct_counts = parsed["punct_counts"]
        function_counts = Counter(w for w in words if w in FUNCTION_WORDS)

        sentence_lengths = [len(re.findall(r"\w+", s)) for s in raw_sents]
        word_lengths = [len(w) for w in words]
        total_punct = sum(punct_counts.values())
        sub_clauses = parsed["sub_clause_count"]
        complex_sents = parsed["complex_sents"]

        open_counts = Counter(parsed["open_pos"])
        n_open = max(1, len(parsed["open_pos"]))

        noun_count = pos_counts["NOUN"]
        verb_count = pos_counts["VERB"]
        dep_depths = parsed["dep_depths"]
        stopword_count = sum(1 for w in words if w in parsed["stopwords"])
        lexical_density_count = pos_counts["NOUN"] + pos_counts["VERB"] + pos_counts["ADJ"] + pos_counts["ADV"]
        conj_count = pos_counts["CONJ"]

    # Compute 116 Features
    features: dict[str, float] = {
        # Metadata counts
        "meta_word_count": float(n_words),
        "meta_sentence_count": float(n_sentences),
        "meta_line_count": float(n_lines),

        # Module A: Lexical
        "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "std_word_length": float(np.std(word_lengths)) if word_lengths else 0.0,
        "ttr": len(set(words)) / n_words if n_words else 0.0,
        "mattr_50": moving_average_ttr(words, 50),
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / n_words if n_words else 0.0,
        "lexical_density": lexical_density_count / n_tokens if n_tokens else 0.0,
        "stopword_rate": stopword_count / n_words if n_words else 0.0,
        "function_word_rate": sum(function_counts.values()) / n_words if n_words else 0.0,

        # Module B: Syntactic
        "noun_rate": pos_counts["NOUN"] / n_tokens if n_tokens else 0.0,
        "verb_rate": pos_counts["VERB"] / n_tokens if n_tokens else 0.0,
        "adj_rate": pos_counts["ADJ"] / n_tokens if n_tokens else 0.0,
        "adv_rate": pos_counts["ADV"] / n_tokens if n_tokens else 0.0,
        "pron_rate": pos_counts["PRON"] / n_tokens if n_tokens else 0.0,
        "adp_rate": pos_counts["ADP"] / n_tokens if n_tokens else 0.0,
        "det_rate": pos_counts["DET"] / n_tokens if n_tokens else 0.0,
        "conj_rate": conj_count / n_tokens if n_tokens else 0.0,
        "noun_to_verb_ratio": noun_count / max(1, verb_count),
        "dependency_depth_mean": float(np.mean(dep_depths)) if dep_depths else 0.0,
        "subordinate_clause_rate": sub_clauses / n_sentences if n_sentences else 0.0,

        # Module C: Structural
        "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "std_sentence_length": float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
        "var_sentence_length": float(np.var(sentence_lengths)) if sentence_lengths else 0.0,
        "avg_line_length": n_words / max(1, n_lines),
        "avg_clause_length": n_words / max(1, n_sentences + sub_clauses),
        "complex_sentence_ratio": complex_sents / n_sentences if n_sentences else 0.0,
        "open_pron_rate": open_counts["PRON"] / n_open,
        "open_det_rate": open_counts["DET"] / n_open,
        "open_conj_rate": (open_counts["CCONJ"] + open_counts["SCONJ"] + open_counts["CONJ"]) / n_open,
        "open_adv_rate": open_counts["ADV"] / n_open,

        # Module D: Punctuation
        "punct_per_token": total_punct / n_tokens if n_tokens else 0.0,
        "comma_rate": punct_counts[","] / total_punct if total_punct else 0.0,
        "semicolon_rate": punct_counts[";"] / total_punct if total_punct else 0.0,
        "colon_rate": punct_counts[":"] / total_punct if total_punct else 0.0,
        "question_rate": punct_counts["?"] / total_punct if total_punct else 0.0,
        "exclamation_rate": punct_counts["!"] / total_punct if total_punct else 0.0,
        "parenthesis_rate": (punct_counts["("] + punct_counts[")"]) / total_punct if total_punct else 0.0,
        "period_rate": punct_counts["."] / total_punct if total_punct else 0.0,
        "quote_rate": (punct_counts['"'] + punct_counts["'"]) / total_punct if total_punct else 0.0,

        # Module E: Character-level linguistic patterns
        "double_letter_rate": sum(1 for w in words if re.search(r"([a-z])\1", w)) / n_words if n_words else 0.0,
        "suffix_ing_rate": sum(1 for w in words if w.endswith("ing")) / n_words if n_words else 0.0,
        "suffix_ed_rate": sum(1 for w in words if w.endswith("ed")) / n_words if n_words else 0.0,
        "suffix_ly_rate": sum(1 for w in words if w.endswith("ly")) / n_words if n_words else 0.0,
        "suffix_tion_rate": sum(1 for w in words if w.endswith("tion")) / n_words if n_words else 0.0,
        "prefix_un_rate": sum(1 for w in words if w.startswith("un") and len(w) > 3) / n_words if n_words else 0.0,
        "prefix_re_rate": sum(1 for w in words if w.startswith("re") and len(w) > 3) / n_words if n_words else 0.0,
        "prefix_in_rate": sum(1 for w in words if w.startswith("in") and len(w) > 3) / n_words if n_words else 0.0,
    }

    # 69 function word individual rates
    for w in FUNCTION_WORDS:
        features[f"fw_{w}"] = function_counts[w] / n_words if n_words else 0.0

    return features


def compute_nlp_module_similarities(
    ref_features: dict[str, float],
    cmp_features: dict[str, float],
    char_ngram_cosine: float | None = None,
) -> dict[str, float]:
    """
    Computes [0, 1] bounded stylistic similarity scores across the 5 NLP modules.
    """
    scores: dict[str, float] = {}
    for module_name, cols in NLP_MODULE_MAP.items():
        diffs = []
        for c in cols:
            a, b = float(ref_features.get(c, 0.0)), float(cmp_features.get(c, 0.0))
            denom = abs(a) + abs(b) + 1e-6
            diffs.append(abs(a - b) / denom)
        sim = float(np.clip(1.0 - np.mean(diffs), 0.0, 1.0))
        if module_name == "character_nlp" and char_ngram_cosine is not None:
            sim = float(0.5 * sim + 0.5 * np.clip(char_ngram_cosine, 0.0, 1.0))
        scores[module_name] = round(sim, 4)
    return scores


def build_nlp_feature_comparison_rows(
    ref_features: dict[str, float],
    cmp_features: dict[str, float],
    word_count_a: int,
    word_count_b: int,
) -> list[dict[str, Any]]:
    """
    Constructs detailed feature comparison rows with absolute & relative difference,
    interpretation, and reliability limitations.
    """
    key_features = [
        "ttr", "mattr_50", "avg_word_length", "hapax_ratio", "function_word_rate",
        "noun_to_verb_ratio", "dependency_depth_mean", "subordinate_clause_rate",
        "avg_sentence_length", "punct_per_token", "comma_rate", "double_letter_rate",
    ]

    rows = []
    min_wc = min(word_count_a, word_count_b)

    for feat in key_features:
        a = float(ref_features.get(feat, 0.0))
        b = float(cmp_features.get(feat, 0.0))
        abs_diff = abs(a - b)
        denom = abs(a) + abs(b) + 1e-6
        rel_diff = abs_diff / denom

        label = FEATURE_HUMAN_LABELS.get(feat, feat)

        # Interpretation generator
        if rel_diff < 0.10:
            interp = f"Close stylistic match ({rel_diff:.1%} variation)."
        elif rel_diff < 0.25:
            interp = f"Moderate stylistic consistency ({rel_diff:.1%} variation)."
        else:
            interp = f"Significant stylistic deviation ({rel_diff:.1%} variation)."

        # Reliability limitation generator
        if min_wc < 20:
            rel_lim = "Low reliability: sample contains under 20 words."
        elif min_wc < 50 and feat in {"mattr_50", "subordinate_clause_rate"}:
            rel_lim = "Moderate reliability: sample size limited for complex syntactic measures."
        else:
            rel_lim = "High reliability: sufficient token evidence."

        rows.append({
            "feature": label,
            "reference": round(a, 4),
            "comparison": round(b, 4),
            "absolute_difference": round(abs_diff, 4),
            "relative_difference": round(rel_diff, 4),
            "interpretation": interp,
            "reliability": rel_lim,
        })

    rows.sort(key=lambda r: r["absolute_difference"], reverse=True)
    return rows
