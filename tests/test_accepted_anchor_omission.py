"""A saved native absence may close a quiet gap between accepted current anchors."""
from __future__ import annotations

from copy import deepcopy

import pytest

from dubsync.asr_crosscheck_config import cross_check_context
from dubsync.missing_dialogue_reconciliation import (
    MissingDialogueEvidence, MissingDialogueQuestion, reconcile_missing_dialogue, reconciliation_context,
)
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, QCFlag, SpeechRegion, TokenMatch, Word
from dubsync.style_profile import StyleProfile


def _case():
    """Captured EP11 441/442 geometry shifted by 1298 s; all hearings are synthetic."""
    source = [Cue(index=i, start_ms=a, end_ms=b, lines=[text]) for i, a, b, text in [
        (437, 1100, 1900, "Boa ideia."), (438, 1880, 2510, "Pode ser."),
        (439, 2510, 3310, "Rápido, rápido. Vem cá."), (440, 3310, 3880, "Chega mais perto."),
        (441, 3880, 4310, "Tao,"), (442, 4310, 5160, "chega mais perto."), (443, 5550, 6240, "Um, dois."),
    ]]
    rows = [("Boa", 1.395, 1.498), ("ideia.", 1.538, 1.895), ("Pode", 2.065, 2.278),
            ("ser.", 2.338, 2.618), ("Rápido,", 2.615, 2.898), ("rápido.", 2.938, 3.058),
            ("Vem", 3.078, 3.079), ("cá,", 3.078, 3.079), ("vem", 3.078, 3.178),
            ("cá.", 3.198, 3.365), ("Mais", 3.525, 3.738), ("perto,", 3.778, 4.065),
            ("mais", 4.175, 4.338), ("perto,", 4.378, 4.618), ("vem.", 4.718, 4.875)]
    words = [Word(text=text, start=a, end=b, speaker_id="primary") for text, a, b in rows]
    secondary = [w.model_copy(update={"speaker_id": "secondary", "confidence": None}) for w in words]
    for i, bounds in {6:(3.12,3.219), 7:(3.24,3.339), 8:(3.36,3.42), 9:(3.48,3.579),
                      10:(3.6,3.74), 11:(3.8,4.08), 12:(4.2,4.319), 13:(4.4,4.639), 14:(4.72,4.92)}.items():
        secondary[i] = secondary[i].model_copy(update={"start": bounds[0], "end": bounds[1]})
    regions = [SpeechRegion(start=a, end=b) for a, b in [(1.395,1.895),(2.065,2.535),(2.615,3.365),(3.525,4.065),(4.175,4.875)]]
    parent = DivergenceSpan(case_id="case-mixed", cue_ids=[441,442], srt_text="Tao chega", asr_text="",
        srt_token_indices=[11,12], start=4.065,end=4.175, left_anchor_cue_id=440,right_anchor_cue_id=442,
        left_anchor_end=4.065,right_anchor_start=4.175)
    left = DivergenceSpan(case_id="case-left",cue_ids=[440],srt_text="Chega",asr_text="vem cá.",
        srt_token_indices=[8],asr_word_indices=[8,9],start=3.078,end=3.365)
    right = DivergenceSpan(case_id="case-right",cue_ids=[442],srt_text="chega",asr_text="",
        srt_token_indices=[12],asr_word_indices=[],start=3.525,end=4.618)
    alignment = AlignmentResult(
        cue_word_indices={437:[0,1],438:[2,3],439:[4,5,6,7],440:[8,9,10,11],441:[],442:[12,13],443:[14]},
        token_matches=[TokenMatch(cue_id=cid,srt_token_index=si,asr_word_index=wi,score=1)
                       for cid,si,wi in [(437,0,0),(437,1,1),(438,2,2),(438,3,3),(439,4,4),(439,5,5),
                                         (439,6,6),(439,7,7),(440,9,10),(440,10,11),(442,13,12),(442,14,13)]],
        divergence_spans=[parent,left,right],unmatched_cue_ids=[441],diagnostics={"missing_audio_cue_ids":[441]},
    )
    current = [c.with_lines(["vem cá. mais perto."]).with_timing(3067,4067) if c.index==440
               else c.with_timing(4067,4310) if c.index==441
               else c.with_lines(["mais perto."]).with_timing(4167,4667) if c.index==442
               else c.with_lines(["vem."]).with_timing(4718,4875) if c.index==443 else c for c in source]
    question = MissingDialogueQuestion(
        span=parent.model_copy(update={"case_id":"missing-dialogue-v1-case-mixed-cue-441", "cue_ids":[441],
            "srt_text":"Tao,", "srt_token_indices":[11], "start":3.525, "end":4.618}),
        parent_case_id="case-mixed",left_word_indices=(10,11),right_word_indices=(12,13),read_only_source_tokens=("chega",),
    )
    def hearing(case_id, text):
        return AdjudicationDecision(case_id=case_id,verdict="use_audio",final_text=text,heard_text=text,
            evidence="heard_clearly",confidence=1,reason="SYNTHETIC TEST FIXTURE ONLY; no native call.")
    ordinary = [hearing("case-left","vem cá."),hearing("case-right","")]
    audio_hash = "a" * 64
    missing = MissingDialogueEvidence([question],reconciliation_context([question],source,alignment,words,regions,audio_sha256=audio_hash),
                                     [hearing(question.span.case_id,"")],[])
    hold=QCFlag(kind="missing_audio_source_cue_held",cue_ids=[441],severity="error",message="Fixture hold.")
    alignment.flags=[hold]
    return {"current_cues":current,"source_cues":source,"alignment":alignment,"words":words,"regions":regions,
        "missing_dialogue":missing,"secondary_words":secondary,
        "secondary_context":cross_check_context(secondary,{"asr":{"provider":"openrouter","model":"microsoft/mai-transcribe-2","language_code":"pt"}}),
        "decisions":ordinary,"verified_audio_sha256":audio_hash,"audio_duration_seconds":10.0,
        "audio_snippet_manifest":{"audio_duration_seconds":10.0,"snippets":[
            {"case_id":caseid,"mime_type":"audio/wav","start":1.525,"end":6.618,"sha256":"b"*64,"size_bytes":163080,"persisted":False}
            for caseid in [question.span.case_id,"case-left","case-right"]]},
        "uncertain_word_indices":set(),"protected_cue_ids":set(),"resolved_cue_ids":set(),
        "flags":[hold,QCFlag(kind="missing_audio_timing_held",cue_ids=[441],message="Fixture timing hold."),
                 QCFlag(kind="timing_evidence_held",cue_ids=[999],message="Unrelated hold.")]}


def _resolve(case):
    from dubsync.accepted_anchor_omission import reconcile_accepted_anchor_omissions
    return reconcile_accepted_anchor_omissions(**case)


def test_existing_guard_restores_false_overlap_but_proven_raw_gap_allows_saved_absence():
    case = _case()
    before = deepcopy(case)
    evidence = case["missing_dialogue"]
    legacy = reconcile_missing_dialogue(case["current_cues"],case["source_cues"],case["alignment"],case["words"],
        case["regions"],evidence.questions,evidence.decisions,StyleProfile(),flags=case["flags"])
    assert legacy.outcomes[0]["outcome"] == "untranscribed_activity_remains"
    assert next(c for c in legacy.cues if c.index==441).end_ms-next(c for c in legacy.cues if c.index==442).start_ms == 143
    result = _resolve(case)
    assert result.resolved_cue_ids == {441}
    assert result.cues == [c for c in case["current_cues"] if c.index != 441]
    assert result.spoken_spans == {}
    assert result.alignment.cue_word_indices == case["alignment"].cue_word_indices
    assert result.alignment.token_matches == case["alignment"].token_matches
    assert result.alignment.diagnostics.missing_audio_cue_ids == []
    assert result.alignment.unmatched_cue_ids == []
    assert [f.kind for f in result.flags] == ["timing_evidence_held","accepted_anchor_omission_reconciled"]
    assert result.flags[0].cue_ids == [999]
    assert result.outcomes[0]["outcome"] == "audio_confirmed_omission"
    proof = result.outcomes[0]["accepted_anchor_omission_proof"]
    assert proof["primary_gap"] == {"start":4.065,"end":4.175}
    assert proof["conservative_gap"] == {"start":4.08,"end":4.175}
    assert proof["raw_regions_in_primary_gap"] == []
    assert len(result.outcomes[0]["accepted_anchor_omission_proof_sha256"]) == 64
    assert case == before


@pytest.mark.parametrize("fault",[
    "tiny_vad_burst","burst_before_intersection","secondary_mismatch","secondary_repetition","secondary_missing","secondary_bad_context",
    "secondary_reversed","secondary_invalid","secondary_gap_closed","primary_placeholder","shared_boundary","uncertain_boundary",
    "changed_boundary","missing_anchor","deleted_source_anchor","source_pair_resolved","protected_target","protected_anchor",
    "source_target_changed","current_target_changed","target_has_words","competing_current_cue","foreign_secondary_word",
    "foreign_primary_word","missing_left_native","missing_right_native","unclear_anchor","deterministic_anchor","wrong_anchor_hearing",
    "partial_edit_wrong_source","partial_edit_wrong_words","duplicate_anchor_decision","overlapping_anchor_edits",
    "missing_absence","unclear_absence","nonempty_absence","non_native_absence","wrong_absence_case","duplicate_absence",
    "missing_manifest","missing_clip","clipped_audio","wrong_clip_case","bad_clip_hash","duplicate_clip","missing_verified_hash",
    "wrong_audio_hash","stale_source_context","stale_question_context","audio_not_required","wrong_question_owner","wrong_question_target",
    "duplicate_question","non_missing_purpose","long_primary_gap","source_pair_deleted_target","provider_failure",
])
def test_uncertain_audio_receipts_words_anchors_or_raw_activity_cannot_remove_cue(fault):
    case=_case(); evidence=case["missing_dialogue"]; q=evidence.questions[0]; a=case["alignment"]
    if fault in {"tiny_vad_burst","burst_before_intersection"}:
        case["regions"].append(SpeechRegion(start=4.10 if fault=="tiny_vad_burst" else 4.068,end=4.105 if fault=="tiny_vad_burst" else 4.075))
        _refresh_context(case)
    elif fault in {"secondary_mismatch","secondary_gap_closed","secondary_invalid"}:
        update={"text":"longe"} if fault=="secondary_mismatch" else {"end":4.25} if fault=="secondary_gap_closed" else {"start":float("nan")}
        case["secondary_words"][11]=case["secondary_words"][11].model_copy(update=update)
        if fault!="secondary_invalid":_refresh_secondary(case)
    elif fault=="secondary_repetition":
        case["secondary_words"].insert(11,Word(text="perto",start=3.9,end=4.05,speaker_id="secondary"));_refresh_secondary(case)
    elif fault=="secondary_missing":case["secondary_words"]=None
    elif fault=="secondary_bad_context":case["secondary_context"]["words_sha256"]="f"*64
    elif fault=="secondary_reversed":
        case["secondary_words"][11],case["secondary_words"][12]=case["secondary_words"][12],case["secondary_words"][11];_refresh_secondary(case)
    elif fault=="primary_placeholder":
        case["words"][11]=case["words"][11].model_copy(update={"start":4.064});_refresh_context(case)
    elif fault=="shared_boundary":a.cue_word_indices[999]=[11]
    elif fault=="uncertain_boundary":case["uncertain_word_indices"]={11}
    elif fault=="changed_boundary":q.span.left_anchor_end=4.0;_refresh_context(case)
    elif fault=="missing_anchor":case["current_cues"]=[c for c in case["current_cues"] if c.index!=440]
    elif fault=="deleted_source_anchor":case["source_cues"]=[c for c in case["source_cues"] if c.index!=440];_refresh_context(case)
    elif fault=="source_pair_resolved":case["resolved_cue_ids"]={442}
    elif fault in {"protected_target","protected_anchor"}:case["protected_cue_ids"]={441 if fault=="protected_target" else 440}
    elif fault=="source_target_changed":case["source_cues"][4]=case["source_cues"][4].with_lines(["Tao vem"]);_refresh_context(case)
    elif fault=="current_target_changed":case["current_cues"][4]=case["current_cues"][4].with_lines(["Ei"])
    elif fault=="target_has_words":a.cue_word_indices[441]=[11]
    elif fault=="competing_current_cue":case["current_cues"].insert(4,Cue(index=999,start_ms=4100,end_ms=4150,lines=["Ei"]))
    elif fault=="foreign_secondary_word":case["secondary_words"].insert(12,Word(text="Tao",start=4.1,end=4.14,speaker_id="secondary"));_refresh_secondary(case)
    elif fault=="foreign_primary_word":case["words"].append(Word(text="Tao",start=4.1,end=4.14,speaker_id="primary"));_refresh_context(case)
    elif fault in {"missing_left_native","missing_right_native"}:case["decisions"].pop(0 if fault=="missing_left_native" else 1)
    elif fault in {"unclear_anchor","deterministic_anchor","wrong_anchor_hearing"}:
        update={"evidence":"heard_unclear"} if fault=="unclear_anchor" else {"reason":"Dual ASR cross-check: exact primary wording corroborated."} if fault=="deterministic_anchor" else {"heard_text":"Chega"}
        case["decisions"][0]=case["decisions"][0].model_copy(update=update)
    elif fault=="partial_edit_wrong_source":a.divergence_spans[1].srt_token_indices=[9]
    elif fault=="partial_edit_wrong_words":a.divergence_spans[1].asr_word_indices=[6,7]
    elif fault=="duplicate_anchor_decision":case["decisions"].append(deepcopy(case["decisions"][0]))
    elif fault=="overlapping_anchor_edits":
        span=deepcopy(a.divergence_spans[1]);span.case_id="extra";a.divergence_spans.append(span)
        case["decisions"].append(case["decisions"][0].model_copy(update={"case_id":"extra"}))
    elif fault=="missing_absence":evidence.decisions.clear()
    elif fault in {"unclear_absence","nonempty_absence","non_native_absence","wrong_absence_case"}:
        update={"evidence":"heard_unclear"} if fault=="unclear_absence" else {"heard_text":"Tao","final_text":"Tao"} if fault=="nonempty_absence" else {"reason":"Dual ASR cross-check: automatic absence."} if fault=="non_native_absence" else {"case_id":"unrelated"}
        evidence.decisions[0]=evidence.decisions[0].model_copy(update=update)
    elif fault=="duplicate_absence":evidence.decisions.append(deepcopy(evidence.decisions[0]))
    elif fault=="missing_manifest":case["audio_snippet_manifest"]=None
    elif fault=="missing_clip":case["audio_snippet_manifest"]["snippets"].pop(0)
    elif fault in {"clipped_audio","wrong_clip_case","bad_clip_hash"}:
        key,value={"clipped_audio":("end",4.5),"wrong_clip_case":("case_id","unrelated"),"bad_clip_hash":("sha256","")}[fault]
        case["audio_snippet_manifest"]["snippets"][0][key]=value
    elif fault=="duplicate_clip":case["audio_snippet_manifest"]["snippets"].append(deepcopy(case["audio_snippet_manifest"]["snippets"][0]))
    elif fault in {"missing_verified_hash","wrong_audio_hash"}:case["verified_audio_sha256"]=None if fault=="missing_verified_hash" else "c"*64
    elif fault in {"stale_source_context","stale_question_context","audio_not_required"}:
        key,value={"stale_source_context":("source_sha256","f"*64),"stale_question_context":("questions_sha256","f"*64),"audio_not_required":("audio_required",False)}[fault]
        evidence.context[key]=value
    elif fault=="wrong_question_owner":q.span.srt_token_indices=[12];_refresh_context(case)
    elif fault=="wrong_question_target":q.span.srt_text="Tao vem";_refresh_context(case)
    elif fault=="duplicate_question":evidence.questions.append(q);_refresh_context(case)
    elif fault=="non_missing_purpose":
        from dataclasses import replace
        evidence.questions[0]=replace(q,purpose="whole_utterance_timing");_refresh_context(case)
    elif fault=="long_primary_gap":q.span.right_anchor_start=4.4;_refresh_context(case)
    elif fault=="source_pair_deleted_target":case["current_cues"]=[c for c in case["current_cues"] if c.index!=441]
    elif fault=="provider_failure":evidence.flags.append(QCFlag(kind="audio_snippet_unavailable",cue_ids=[441],message="Fixture missing audio."))
    before=deepcopy(case)
    result=_resolve(case)
    assert not result.resolved_cue_ids and result.cues==case["current_cues"] and not result.spoken_spans
    assert case==before


def _refresh_context(case):
    evidence=case["missing_dialogue"]
    evidence.context.update(reconciliation_context(evidence.questions,case["source_cues"],case["alignment"],case["words"],case["regions"],audio_sha256="a"*64))


def _refresh_secondary(case):
    case["secondary_context"]=cross_check_context(case["secondary_words"],{"asr":{"provider":"openrouter","model":"microsoft/mai-transcribe-2","language_code":"pt"}})
