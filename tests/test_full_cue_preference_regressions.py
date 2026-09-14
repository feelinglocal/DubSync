from __future__ import annotations

from dubsync import aligner
from dubsync.models import Cue, Word
from dubsync.tokenize import normalized_words, tokenize_cues


def test_short_full_cue_does_not_steal_later_mai_speech_from_retained_phrase():
    # Frozen original-source and MAI evidence from episode 17, 848–860 seconds.
    source = [
        (211, 848640, 849520, "Vamos denunciar."),
        (212, 851240, 851830, "Não pode!"),
        (213, 852070, 853190, "Não pode de jeito nenhum fazer denúncia."),
        (214, 853670, 854830, "As fotos ainda estão com ele,"),
        (215, 854830, 855670, "não pode denunciar."),
        (216, 855880, 857640, "Se ele divulgar essas fotos,"),
        (217, 857670, 859240, "eu vou estar ferrada de vez."),
    ]
    cues = [Cue(index=index, start_ms=start, end_ms=end, lines=[text])
            for index, start, end, text in source]
    evidence = [
        ("Vamos", 848.64, 848.859, "A"), ("denunciar.", 848.96, 849.52, "A"),
        ("Não,", 851.32, 851.559, "B"), ("de", 852.16, 852.259, "B"),
        ("jeito", 852.32, 852.539, "B"), ("nenhum.", 852.60, 852.959, "B"),
        ("A", 853.84, 853.92, "B"), ("gente", 854.00, 854.179, "B"),
        ("não", 854.24, 854.36, "B"), ("pode,", 854.44, 854.64, "B"),
        ("ele", 854.68, 854.78, "B"), ("tá", 854.84, 854.94, "B"),
        ("com", 854.96, 855.04, "B"), ("as", 855.08, 855.14, "B"),
        ("fotos", 855.20, 855.48, "B"), ("ainda.", 855.60, 855.879, "B"),
        ("E", 856.36, 856.42, "B"), ("se", 856.44, 856.54, "B"),
        ("ele", 856.58, 856.679, "B"), ("divulgar", 856.76, 857.06, "B"),
        ("essas", 857.08, 857.32, "B"), ("fotos,", 857.40, 857.779, "B"),
        ("eu", 858.00, 858.08, "B"), ("tô", 858.12, 858.239, "B"),
        ("ferrada.", 858.36, 858.96, "B"),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None, speaker_id=speaker)
             for text, start, end, speaker in evidence]

    result = aligner.align_cues_to_words(cues, words)

    assert result.cue_word_indices[212] == [2]
    assert result.cue_word_indices[213] == [3, 4, 5]
    assert len(result.token_matches) == 16
    assert result.diagnostics.unresolved is False


def test_unrelated_full_cue_recovery_cannot_subsidize_lost_neighbor_anchors():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1000, lines=["Não pode!"]),
        Cue(index=2, start_ms=1000, end_ms=2000, lines=["de jeito nenhum fazer denúncia"]),
        Cue(index=3, start_ms=3000, end_ms=4000, lines=["alpha beta gamma"]),
    ]
    words = [Word(text=text, start=index, end=index + .5)
             for index, text in enumerate("Não de jeito nenhum não pode alpha beta gamma".split())]
    tokens = tokenize_cues(cues)
    initial_pairs = {(0, 0), (2, 1), (3, 2), (4, 3)}
    initial_ops = aligner._ops_from_exact_pairs(len(tokens), len(words), initial_pairs)

    result_ops = aligner._prefer_unique_full_cue_windows(initial_ops, tokens, normalized_words(words))

    result_pairs = {(op.srt_index, op.asr_index) for op in result_ops if op.kind == "match"}
    assert initial_pairs <= result_pairs
    assert {(7, 6), (8, 7), (9, 8)} <= result_pairs
    assert (0, 4) not in result_pairs


def test_full_cue_recovery_preserves_an_already_complete_neighbor():
    # Original cues 78/79 and actual full-episode MAI words. The repeated
    # mention later in the episode means only cue 79 has a unique full window.
    cues = [
        Cue(index=78, start_ms=252400, end_ms=252750, lines=["A Lumi"]),
        Cue(index=79, start_ms=252750, end_ms=254630,
            lines=["A Lumi foi atrás daquelas duas despesas"]),
    ]
    evidence = [
        ("A", 252.12, 252.199), ("Lumi", 252.28, 252.56),
        ("foi", 252.6, 252.74), ("atrás", 252.76, 253.0),
        ("daquelas", 253.02, 253.379), ("duas", 253.44, 253.62),
        ("despesas", 253.72, 254.28),
        ("a", 285.22, 285.239), ("Lumi", 285.279, 285.479),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in evidence]
    tokens = tokenize_cues(cues)
    initial_pairs = {(0, 0), (1, 1), *[(index + 2, index) for index in range(2, 7)]}
    initial_ops = aligner._ops_from_exact_pairs(len(tokens), len(words), initial_pairs)

    result_ops = aligner._prefer_unique_full_cue_windows(initial_ops, tokens, normalized_words(words))

    result_pairs = {(op.srt_index, op.asr_index) for op in result_ops if op.kind == "match"}
    assert result_pairs == initial_pairs
