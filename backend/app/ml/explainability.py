"""
explainability.py — Educational concepts registry and evidence explanations.

Covers curated concepts:
1. OCR (Optical Character Recognition)
2. Tokenization & Segmentation
3. Part-of-Speech (POS) Tagging
4. Lexical Diversity (TTR, MATTR, Hapax)
5. Function Words Analysis
6. Character N-Grams
7. Dependency Parsing & Syntactic Depth
8. Stylometry & Linguistic Fingerprinting
9. CNN / Visual Feature Extraction
10. Pairwise Verification Classification
11. Model Evaluation Metrics (EER, FAR, FRR, ROC-AUC)

Never fabricates explainability outputs.
"""
from __future__ import annotations

from typing import Any
from app.schemas.verification import FeatureExplanation, LearningModuleItem


# Curated educational concept registry
EDUCATIONAL_MODULES: dict[str, dict[str, Any]] = {
    "ocr": {
        "id": "ocr",
        "title": "Optical Character Recognition (TrOCR)",
        "category": "Vision-Language Models",
        "summary": "Transformer-based sequence-to-sequence line transcription from handwritten images.",
        "detailed_explanation": (
            "TrOCR couples an image Vision Transformer (ViT) encoder with a language Transformer decoder. "
            "Unlike traditional CNN-RNN-CTC pipelines, TrOCR captures long-range bidirectional visual context "
            "and generates transcriptions autoregressively. In Check Mate, documents are first segmented into "
            "text lines to maintain sequence fidelity and avoid distortion."
        ),
        "mathematical_formulation": "P(Y|X) = \\prod_{t=1}^{T} P(y_t | y_{<t}, \\text{Encoder}(X))",
        "relevance_to_authorship": (
            "Accurate transcription enables downstream NLP stylometric analysis. However, OCR transcription errors "
            "can propagate into lexical and syntactic feature values, requiring error-tolerant representations."
        ),
        "limitations": "Heavily cursive, overlapping, or severely degraded lines can induce character substitutions or dropouts.",
    },
    "tokenization": {
        "id": "tokenization",
        "title": "Tokenization & Text Segmentation",
        "category": "Natural Language Processing",
        "summary": "Splitting continuous text into discrete units (tokens, words, punctuation, and sentences).",
        "detailed_explanation": (
            "Tokenization establishes the fundamental units of linguistic observation. Check Mate segments "
            "text into words, punctuation tokens, and sentences, isolating contractions, affixes, and boundaries."
        ),
        "mathematical_formulation": "\\text{Text} \\to [t_1, t_2, \\dots, t_N], \\quad t_i \\in \\mathcal{V}",
        "relevance_to_authorship": "Consistent token boundaries are essential for calculating lexical richness, word lengths, and syntactic structures.",
        "limitations": "Irregular punctuation or lack of capitalization in handwritten transcriptions can alter sentence boundary detection.",
    },
    "pos_tagging": {
        "id": "pos_tagging",
        "title": "Part-of-Speech (POS) Tagging",
        "category": "Syntactic Analysis",
        "summary": "Grammatical category labeling (nouns, verbs, adjectives, adverbs, prepositions, determiners).",
        "detailed_explanation": (
            "POS tagging assigns universal syntactic categories to tokens in context. An author's subconscious "
            "ratio of nominal versus verbal phrasing (noun-to-verb ratio) and use of modifying adverbs/adjectives "
            "acts as an involuntary stylistic signature."
        ),
        "mathematical_formulation": "\\hat{T} = \\arg\\max_T P(T | W)",
        "relevance_to_authorship": "Syntactic habits are content-invariant and difficult for an author to consciously disguise.",
        "limitations": "Short texts provide insufficient counts for fine-grained POS distribution estimation.",
    },
    "lexical_diversity": {
        "id": "lexical_diversity",
        "title": "Lexical Diversity (TTR, MATTR, Hapax)",
        "category": "Stylometry",
        "summary": "Quantification of vocabulary richness and repetition tendencies.",
        "detailed_explanation": (
            "Standard Type-Token Ratio (TTR = V / N) suffers from severe text-length dependency: longer texts "
            "naturally repeat words, depressing TTR. Moving-Average TTR (MATTR) resolves this by sliding a fixed window "
            "(e.g., 50 words) over the text to compute a length-invariant vocabulary richness score. Hapax legomena "
            "measures the proportion of words appearing exactly once."
        ),
        "mathematical_formulation": "\\text{TTR} = \\frac{|V|}{N}, \\quad \\text{MATTR}_w = \\frac{1}{K} \\sum_{k=1}^K \\frac{|V(w_k)|}{w}",
        "relevance_to_authorship": "Reflects an author's active lexicon size and stylistic tendency toward lexical variety.",
        "limitations": "Unreliable on samples under 20 words where token counts are statistically unrepresentative.",
    },
    "function_words": {
        "id": "function_words",
        "title": "Function Word Analysis",
        "category": "Stylometry",
        "summary": "Frequency profiles of closed-class grammatical words (pronouns, prepositions, conjunctions, auxiliary verbs).",
        "detailed_explanation": (
            "Function words ('the', 'of', 'and', 'to', 'which', 'although') convey grammatical structure rather "
            "than thematic content. Decades of stylometric research (Mosteller & Wallace) demonstrate that authors "
            "exhibit idiosyncratic, subconscious frequencies of function word usage that remain stable across topics."
        ),
        "mathematical_formulation": "f(w_i) = \\frac{\\text{count}(w_i)}{\\sum_j \\text{count}(w_j)}",
        "relevance_to_authorship": "Primary signal for topic-independent authorship attribution.",
        "limitations": "Requires adequate document length (typically 50+ words) for stable frequency estimates.",
    },
    "character_ngrams": {
        "id": "character_ngrams",
        "title": "Character N-Grams & Morphological Patterns",
        "category": "Subword NLP",
        "summary": "Sub-word character sequence distributions capturing spelling habits, prefixes, and suffixes.",
        "detailed_explanation": (
            "Character 3-grams, 4-grams, and 5-grams capture morphological affixes ('-ing', '-tion', 'un-'), punctuation "
            "co-occurrences, and idiosyncratic subword patterns. Check Mate applies TF-IDF weighting and cosine similarity "
            "to quantify sub-word lexical alignment between documents."
        ),
        "mathematical_formulation": "\\text{CosineSimilarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}",
        "relevance_to_authorship": "Highly resilient to OCR errors and spelling variations compared to whole-word models.",
        "limitations": "Sensitive to topic vocabulary when text samples describe vastly different domains.",
    },
    "dependency_parsing": {
        "id": "dependency_parsing",
        "title": "Syntactic Dependency Parsing",
        "category": "Deep Syntactic Analysis",
        "summary": "Grammatical tree structures and syntactic subordination analysis.",
        "detailed_explanation": (
            "Dependency parsing analyzes head-dependent relationships between words. Tree depth measures sentence "
            "structural complexity, while subordinate clause frequency ('advcl', 'relcl', 'ccomp') reflects whether "
            "the author favors hypotactic (nested) or paratactic (flat, coordinated) prose style."
        ),
        "mathematical_formulation": "\\text{Depth}(T) = \\max_{t \\in T} \\text{dist}(t, \\text{ROOT})",
        "relevance_to_authorship": "Syntactic branching habits are deeply ingrained and immune to conscious manipulation.",
        "limitations": "Requires grammatically coherent complete sentences; poor handwriting transcription degrades parse trees.",
    },
    "stylometry": {
        "id": "stylometry",
        "title": "Linguistic Fingerprinting & Stylometry",
        "category": "Authorship Science",
        "summary": "Multi-dimensional statistical characterization of personal writing style.",
        "detailed_explanation": (
            "Stylometry treats an individual's writing as a multi-dimensional feature vector spanning lexical richness, "
            "syntax, punctuation pacing, and structural habits. Check Mate extracts a 116-dimensional linguistic vector "
            "for each text and verifies authorship through pairwise vector differences."
        ),
        "mathematical_formulation": "\\Delta(\\mathbf{x}_A, \\mathbf{x}_B) = |\\mathbf{x}_A - \\mathbf{x}_B|",
        "relevance_to_authorship": "Enables quantitative authorship verification based on objective linguistic traits.",
        "limitations": "Does not account for ghostwriting, direct transcription of dictated text, or intentional imitation.",
    },
    "visual_features": {
        "id": "visual_features",
        "title": "Computer Vision Handwriting Features",
        "category": "Physical Graphology & CV",
        "summary": "Handcrafted physical measurements: stroke width, pressure, slant angle, and letter geometry.",
        "detailed_explanation": (
            "Using OpenCV, Check Mate extracts 8 normalized physical handwriting features: ink density inside the text "
            "bounding box, stroke width via Euclidean distance transform, stroke width variation (pen pressure dynamics), "
            "Sobel gradient slant angle, slant consistency, letter aspect ratios, and letter height uniformity."
        ),
        "mathematical_formulation": "\\theta = \\arctan\\left(\\frac{\\partial I / \\partial y}{\\partial I / \\partial x}\\right)",
        "relevance_to_authorship": "Independent of the linguistic message, capturing the neuromuscular biometric habits of the writer.",
        "limitations": "Variations in writing instrument (fountain pen vs ballpoint) and writing posture can shift physical measurements.",
    },
    "pairwise_classification": {
        "id": "pairwise_classification",
        "title": "Pairwise Verification vs Closed-Set Attribution",
        "category": "Machine Learning",
        "summary": "Formulating authorship as a 1-vs-1 compatibility decision between unseen writers.",
        "detailed_explanation": (
            "Unlike closed-set attribution (which picks an author from a fixed set of known candidates), pairwise verification "
            "evaluates whether two arbitrary documents share the same author (y=1) or different authors (y=0). "
            "Training utilizes writer-disjoint splits to guarantee that the classifier learns generalizable style differences "
            "rather than memorizing specific writers."
        ),
        "mathematical_formulation": "f(\\mathbf{x}_A, \\mathbf{x}_B) \\to [0, 1] \\quad \\text{where } y \\in \\{0, 1\\}",
        "relevance_to_authorship": "Standard formulation for forensic comparison, signature verification, and educational integrity.",
        "limitations": "Requires careful balancing of positive and negative training pairs to avoid prior-probability bias.",
    },
    "evaluation_metrics": {
        "id": "evaluation_metrics",
        "title": "Biometric & Verification Evaluation Metrics",
        "category": "Evaluation Science",
        "summary": "Rigorous benchmarking: Accuracy, ROC-AUC, EER, False Acceptance Rate (FAR), and False Rejection Rate (FRR).",
        "detailed_explanation": (
            "In verification systems, accuracy alone is insufficient due to class imbalance. Check Mate evaluates: "
            "FAR (rate at which different writers are falsely accepted as the same), FRR (rate at which true authors are rejected), "
            "and Equal Error Rate (EER, the operating point where FAR = FRR). The threshold is calibrated on validation data."
        ),
        "mathematical_formulation": "\\text{FAR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}, \\quad \\text{FRR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TP}}, \\quad \\text{EER} = \\text{FAR}(\\theta^*) = \\text{FRR}(\\theta^*)",
        "relevance_to_authorship": "Guarantees scientifically honest performance reporting without misleading accuracy figures.",
        "limitations": "Operating thresholds must be recalibrated if the deployment domain differs substantially from training data.",
    },
}


FEATURE_METADATA_REGISTRY: dict[str, dict[str, str]] = {
    "Ink Packing Density (Local)": {
        "definition": "Proportion of active ink pixels relative to the local stroke corridor envelope, invariant to page margins.",
        "formula": "\\text{InkDensity} = \\frac{|\\text{InkPixels}|}{\\text{Area}(\\text{DilationEnvelope})}",
        "module": "visual",
    },
    "Ink Packing Density": {
        "definition": "Proportion of active ink pixels relative to handwriting regions.",
        "formula": "\\text{InkDensity} = \\frac{|\\text{InkPixels}|}{\\text{Area}(\\text{Envelope})}",
        "module": "visual",
    },
    "Inter-Character Spacing": {
        "definition": "Median horizontal whitespace gap between consecutive letters along text lines, normalized by character width.",
        "formula": "\\text{SpacingRatio} = \\text{median}\\left(\\frac{x_{i+1} - (x_i + w_i)}{w_i}\\right)",
        "module": "visual",
    },
    "Baseline Consistency": {
        "definition": "Standard deviation of character component bottom boundaries along text lines, normalized by letter height.",
        "formula": "\\sigma_{\\text{base}} = \\frac{\\text{std}(y_i + h_i)}{\\mu_h}",
        "module": "visual",
    },
    "Stroke Width (normalized)": {
        "definition": "Mean physical stroke thickness estimated via Euclidean distance transform at canonical 32px letter scale.",
        "formula": "w = 2.0 \\times \\mathbb{E}[\\text{DistTransform}(\\text{Binary})]",
        "module": "visual",
    },
    "Stroke Width Variation": {
        "definition": "Standard deviation of stroke width across letter strokes, reflecting pen pressure modulation.",
        "formula": "\\sigma_w = 2.0 \\times \\text{std}(\\text{DistTransform}(\\text{Binary}))",
        "module": "visual",
    },
    "Slant Angle (avg)": {
        "definition": "Mean angular orientation of stroke edges computed from Sobel spatial image gradients.",
        "formula": "\\theta = \\text{mean}(\\arctan2(G_y, G_x) + 90^\\circ \\pmod{180^\\circ} - 90^\\circ)",
        "module": "visual",
    },
    "Slant Consistency": {
        "definition": "Standard deviation of stroke slant angles; smaller values indicate disciplined, uniform handwriting slant.",
        "formula": "\\sigma_\\theta = \\text{std}(\\theta_i)",
        "module": "visual",
    },
    "Letter Width/Height Ratio": {
        "definition": "Mean bounding-box aspect ratio across connected letter components.",
        "formula": "\\text{Aspect} = \\mathbb{E}[w_{\\text{cc}} / h_{\\text{cc}}]",
        "module": "visual",
    },
    "Letter Height Uniformity": {
        "definition": "Relative variation (coefficient of variation) in connected-component letter heights.",
        "formula": "\\text{RelStd} = \\frac{\\sigma_{h}}{\\mu_{h}}",
        "module": "visual",
    },
    "Stroke-to-Letter Ratio": {
        "definition": "Ratio of stroke thickness to letter height, distinguishing bold/heavy hands from delicate scripts.",
        "formula": "\\text{StrokeRatio} = \\frac{\\text{StrokeWidth}}{\\mu_{h}}",
        "module": "visual",
    },
    "Type-Token Ratio (Vocabulary Richness)": {
        "definition": "Ratio of unique word types to total word tokens; measures active vocabulary diversity.",
        "formula": "\\text{TTR} = \\frac{|V|}{N}",
        "module": "lexical",
    },
    "Moving-Average TTR (Window 50)": {
        "definition": "Length-invariant vocabulary richness computed over rolling 50-word windows.",
        "formula": "\\text{MATTR}_{50} = \\frac{1}{K}\\sum_{k=1}^K \\frac{|V_k|}{50}",
        "module": "lexical",
    },
    "Average Word Length": {
        "definition": "Mean character count per word, reflecting linguistic register and vocabulary complexity.",
        "formula": "\\bar{L}_w = \\frac{1}{N}\\sum_{i=1}^N \\text{len}(w_i)",
        "module": "lexical",
    },
    "Hapax Legomena Ratio (Unique Words)": {
        "definition": "Proportion of vocabulary words that appear exactly once in the document.",
        "formula": "\\text{HapaxRatio} = \\frac{\\sum [\\text{count}(w)=1]}{|V|}",
        "module": "lexical",
    },
    "Function Word Usage Rate": {
        "definition": "Aggregate proportion of tokens belonging to standard closed-class function word categories.",
        "formula": "\\text{FWRate} = \\frac{\\sum_{w \\in \\mathcal{F}} \\text{count}(w)}{N}",
        "module": "lexical",
    },
    "Noun-to-Verb Ratio": {
        "definition": "Ratio of nouns to verbs, distinguishing nominal writing styles from verbal/action-oriented prose.",
        "formula": "\\text{NVRatio} = \\frac{\\text{count}(\\text{NOUN}) + 1}{\\text{count}(\\text{VERB}) + 1}",
        "module": "syntactic",
    },
    "Syntactic Tree Depth": {
        "definition": "Mean maximum depth of dependency parse trees across all sentences in the sample.",
        "formula": "\\bar{D} = \\frac{1}{|S|}\\sum_{s \\in S} \\max_{t \\in s} \\text{depth}(t)",
        "module": "syntactic",
    },
    "Subordinate Clause Density": {
        "definition": "Average number of subordinate and relative clauses per sentence.",
        "formula": "\\text{SubRate} = \\frac{|\\text{SubClauses}|}{|S|}",
        "module": "syntactic",
    },
    "Average Sentence Length (Words)": {
        "definition": "Mean word count per sentence; a fundamental marker of syntactic pacing and structural rhythm.",
        "formula": "\\bar{L}_s = \\frac{N}{|S|}",
        "module": "structural",
    },
    "Punctuation Density": {
        "definition": "Total punctuation mark frequency relative to overall token count.",
        "formula": "\\text{PunctRate} = \\frac{|\\text{PunctTokens}|}{|\\text{AllTokens}|}",
        "module": "punctuation",
    },
    "Comma Relative Frequency": {
        "definition": "Frequency of commas relative to total punctuation occurrences.",
        "formula": "\\text{CommaRate} = \\frac{\\text{count}(',')}{|\\text{PunctTokens}|}",
        "module": "punctuation",
    },
    "Double Letter Frequency": {
        "definition": "Rate of words containing consecutive identical letters (e.g. 'letter', 'book').",
        "formula": "\\text{DoubleRate} = \\frac{\\sum [\\text{has\\_double}(w)]}{N}",
        "module": "character_nlp",
    },
}


def build_feature_explanations(
    visual_rows: list[dict[str, Any]],
    nlp_rows: list[dict[str, Any]],
) -> list[FeatureExplanation]:
    """
    Constructs comprehensive educational explanations for all compared features.
    """
    explanations: list[FeatureExplanation] = []
    combined_rows = visual_rows + nlp_rows

    for row in combined_rows:
        feat_name = row.get("feature", "")
        meta = FEATURE_METADATA_REGISTRY.get(feat_name, {})

        definition = meta.get(
            "definition",
            f"Stylometric or physical measurement comparing handwriting characteristics."
        )
        formula = meta.get("formula")
        module = meta.get("module", "general")

        ref_val = row.get("reference")
        cmp_val = row.get("comparison")
        diff_val = row.get("absolute_difference")
        interp = row.get("interpretation", "Evidence comparison between reference and comparison.")
        reliability = row.get("reliability", "Standard feature measurement.")

        explanations.append(FeatureExplanation(
            feature=feat_name,
            name=feat_name,
            module=module,
            definition=definition,
            formula=formula,
            reference_value=ref_val,
            comparison_value=cmp_val,
            difference=diff_val,
            interpretation=interp,
            reliability_limitations=reliability,
        ))

    return explanations


def get_all_learning_modules() -> list[LearningModuleItem]:
    """Returns all curated educational learning modules."""
    return [
        LearningModuleItem(**data)
        for data in EDUCATIONAL_MODULES.values()
    ]


def get_learning_module_by_id(module_id: str) -> LearningModuleItem | None:
    """Returns a specific educational learning module by ID."""
    clean_id = module_id.lower().replace("-", "_")
    data = EDUCATIONAL_MODULES.get(clean_id)
    if data:
        return LearningModuleItem(**data)
    return None
