configfile: "config.yaml"

DATA = config["data_dir"]
RES  = config["results_dir"]


rule all:
    input:
        # Descriptive (Section 5)
        f"{RES}/figures/vowel_chart.png",
        f"{RES}/figures/boxplots_f1_f2.png",
        f"{RES}/figures/violin_intraspeaker.png",
        f"{RES}/figures/neural_proj_phoneme.png",
        f"{RES}/figures/neural_proj_L1.png",
        f"{RES}/figures/neural_proj_gender.png",
        f"{RES}/tables/descriptive_stats.csv",
        f"{RES}/tables/variance_decomposition.csv",
        f"{RES}/tables/mantel_rsm.csv",
        f"{RES}/tables/neural_variance_ratio.csv",
        f"{RES}/tables/neural_similarity_ratio.csv",
        f"{RES}/tables/neural_variance_decomposition.csv",
        # Missing values report (Section 4)
        f"{RES}/tables/missing_values.csv",
        # Statistical tests (Section 6)
        f"{RES}/tables/group_comparisons.csv",
        f"{RES}/tables/gender_residual.csv",
        f"{RES}/tables/distance_matrices_mantel.csv",
        f"{RES}/tables/pairwise_distance_ci.csv",
        f"{RES}/tables/classifier_results.csv",
        f"{RES}/tables/mcnemar_results.csv",
        f"{RES}/figures/confusion_matrices.png",
        # LME (Section 7)
        f"{RES}/tables/lme_results.csv",
        # CIs + ROPE (Section 8)
        f"{RES}/figures/forest_plots.png",
        f"{RES}/tables/rope_classification.csv",
        # Clustering (Section 9)
        f"{RES}/figures/dendrograms.png",
        f"{RES}/tables/ari_scores.csv",
        f"{RES}/tables/consonant_vowel_ari.csv",


# ── Stage 1: parse corpus ────────────────────────────────────────────────────

rule parse_corpus:
    input:
        metadata = config["metadata_file"],
        rufrcorr = config["rufrcorr_file"],
    output:
        phonemes = f"{DATA}/phonemes.csv",
    params:
        corpus_dir = config["corpus_dir"],
    script:
        "pipeline/parse_corpus.py"


# ── Stage 2: acoustic features ───────────────────────────────────────────────

rule extract_acoustics:
    input:
        phonemes = f"{DATA}/phonemes.csv",
    output:
        features       = f"{DATA}/features_acoustic.csv",
        missing_values = f"{RES}/tables/missing_values.csv",
    params:
        max_formant_female  = config["max_formant_female"],
        max_formant_male    = config["max_formant_male"],
        n_formants          = config["n_formants"],
        long_vowel_ms       = config["long_vowel_threshold_ms"],
    script:
        "pipeline/extract_acoustics.py"


# ── Stage 3: Whisper neural features ─────────────────────────────────────────

rule extract_neural_whisper:
    input:
        phonemes = f"{DATA}/phonemes.csv",
    output:
        features = f"{DATA}/features_whisper.npz",
    params:
        model        = config["whisper_model"],
        layer_lower  = config["whisper_layer_lower"],
        layer_upper  = config["whisper_layer_upper"],
    script:
        "pipeline/extract_neural_whisper.py"


# ── Stage 4: XLS-R neural features ───────────────────────────────────────────

rule extract_neural_xlsr:
    input:
        phonemes = f"{DATA}/phonemes.csv",
    output:
        features = f"{DATA}/features_xlsr.npz",
    params:
        model  = config["xlsr_model"],
        layers = config["xlsr_layers"],
    script:
        "pipeline/extract_neural_xlsr.py"


# ── Stage 5: normalise ────────────────────────────────────────────────────────

rule normalise:
    input:
        acoustic = f"{DATA}/features_acoustic.csv",
        whisper  = f"{DATA}/features_whisper.npz",
        xlsr     = f"{DATA}/features_xlsr.npz",
    output:
        acoustic_norm = f"{DATA}/features_acoustic_norm.csv",
        whisper_pca   = f"{DATA}/features_whisper_pca.npz",
        xlsr_pca      = f"{DATA}/features_xlsr_pca.npz",
    params:
        pca_analysis  = config["pca_n_components_analysis"],
        pca_viz       = config["pca_n_components_viz"],
        umap_neighbors = config["umap_n_neighbors"],
        umap_min_dist  = config["umap_min_dist"],
        seed           = config["random_seed"],
    script:
        "pipeline/normalise.py"


# ── Stage 6a: descriptive statistics ─────────────────────────────────────────

rule descriptive:
    input:
        acoustic_norm = f"{DATA}/features_acoustic_norm.csv",
        whisper_pca   = f"{DATA}/features_whisper_pca.npz",
        xlsr_pca      = f"{DATA}/features_xlsr_pca.npz",
    output:
        vowel_chart           = f"{RES}/figures/vowel_chart.png",
        boxplots_f1_f2        = f"{RES}/figures/boxplots_f1_f2.png",
        violin_intraspeaker   = f"{RES}/figures/violin_intraspeaker.png",
        neural_proj_phoneme   = f"{RES}/figures/neural_proj_phoneme.png",
        neural_proj_l1        = f"{RES}/figures/neural_proj_L1.png",
        neural_proj_gender    = f"{RES}/figures/neural_proj_gender.png",
        descriptive_stats     = f"{RES}/tables/descriptive_stats.csv",
        variance_decomp       = f"{RES}/tables/variance_decomposition.csv",
        mantel_rsm            = f"{RES}/tables/mantel_rsm.csv",
        neural_variance_ratio = f"{RES}/tables/neural_variance_ratio.csv",
        neural_similarity_ratio = f"{RES}/tables/neural_similarity_ratio.csv",
        neural_variance_decomp = f"{RES}/tables/neural_variance_decomposition.csv",
    params:
        french_vowels = config["french_oral_vowels"],
        seed          = config["random_seed"],
    script:
        "analysis/descriptive.py"


# ── Stage 6b: statistical tests ──────────────────────────────────────────────

rule stat_tests:
    input:
        acoustic_norm = f"{DATA}/features_acoustic_norm.csv",
        whisper_pca   = f"{DATA}/features_whisper_pca.npz",
        xlsr_pca      = f"{DATA}/features_xlsr_pca.npz",
    output:
        group_comparisons     = f"{RES}/tables/group_comparisons.csv",
        gender_residual       = f"{RES}/tables/gender_residual.csv",
        distance_mantel       = f"{RES}/tables/distance_matrices_mantel.csv",
        pairwise_distance_ci  = f"{RES}/tables/pairwise_distance_ci.csv",
        classifier_results    = f"{RES}/tables/classifier_results.csv",
        mcnemar_results       = f"{RES}/tables/mcnemar_results.csv",
        confusion_matrices    = f"{RES}/figures/confusion_matrices.png",
    params:
        n_permutations = config["n_permutations"],
        bootstrap_n    = config["bootstrap_n"],
        french_vowels  = config["french_oral_vowels"],
        seed           = config["random_seed"],
    script:
        "analysis/stat_tests.py"


# ── Stage 6c: linear mixed-effects models ────────────────────────────────────

rule mixed_effects:
    input:
        acoustic_norm = f"{DATA}/features_acoustic_norm.csv",
        whisper_pca   = f"{DATA}/features_whisper_pca.npz",
        xlsr_pca      = f"{DATA}/features_xlsr_pca.npz",
    output:
        lme_results = f"{RES}/tables/lme_results.csv",
    params:
        french_vowels = config["french_oral_vowels"],
        seed          = config["random_seed"],
    script:
        "analysis/mixed_effects.py"


# ── Stage 6d: confidence intervals + ROPE ────────────────────────────────────

rule confidence_rope:
    input:
        acoustic_norm = f"{DATA}/features_acoustic_norm.csv",
        whisper_pca   = f"{DATA}/features_whisper_pca.npz",
        xlsr_pca      = f"{DATA}/features_xlsr_pca.npz",
        lme_results   = f"{RES}/tables/lme_results.csv",
    output:
        forest_plots       = f"{RES}/figures/forest_plots.png",
        rope_classification = f"{RES}/tables/rope_classification.csv",
    params:
        bootstrap_n    = config["bootstrap_n"],
        rope_hz        = config["rope_acoustic_hz"],
        french_vowels  = config["french_oral_vowels"],
        seed           = config["random_seed"],
    script:
        "analysis/confidence_rope.py"


# ── Stage 6e: hierarchical clustering ────────────────────────────────────────

rule clustering:
    input:
        acoustic_norm = f"{DATA}/features_acoustic_norm.csv",
        whisper_pca   = f"{DATA}/features_whisper_pca.npz",
        xlsr_pca      = f"{DATA}/features_xlsr_pca.npz",
    output:
        dendrograms          = f"{RES}/figures/dendrograms.png",
        ari_scores           = f"{RES}/tables/ari_scores.csv",
        consonant_vowel_ari  = f"{RES}/tables/consonant_vowel_ari.csv",
    params:
        french_vowels = config["french_oral_vowels"],
        seed          = config["random_seed"],
    script:
        "analysis/clustering.py"
