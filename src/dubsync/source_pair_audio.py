"""Supply a complete context clip and a separate exact candidate excerpt."""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock
import wave

from .audio_snippets import AudioSnippetError, extract_audio_snippets
from .providers import ProviderError


SOURCE_PAIR_CANDIDATE_AUDIO_POLICY_VERSION = 1


class SourcePairAudioAdapter:
    def __init__(self, adapter, questions, audio_path, directory, *, extractor=extract_audio_snippets):
        self.adapter = adapter
        self.questions = {question.span.case_id: question for question in questions}
        self.audio_path, self.directory = Path(audio_path), Path(directory)
        self.extractor = extractor
        self._records = {}
        self._lock = RLock()

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def manifest(self):
        with self._lock:
            return [dict(self._records[key]) for key in sorted(self._records)]

    def adjudicate_with_audio(self, spans, snippets):
        method = getattr(self.adapter, "adjudicate_with_audio", None)
        if not callable(method):
            raise ProviderError("Source pair candidate hearing requires an audio-capable provider.")
        candidates = []
        for span in spans:
            question = self.questions.get(span.case_id)
            if question is None:
                continue
            if question.span.model_dump() != span.model_dump():
                raise ProviderError("Source pair candidate does not match its question.")
            candidates.append(span.model_copy(update={
                "case_id": span.case_id + "-candidate",
                "start": question.utterance_start_seconds, "end": question.utterance_end_seconds,
            }))
        try:
            clips = self.extractor(self.audio_path, candidates, self.directory, pad_seconds=0,
                                   max_duration_seconds=16, max_covering_duration_seconds=16) if candidates else []
            by_id = {clip.case_id: clip for clip in clips}
            if len(by_id) != len(candidates) or len(clips) != len(candidates):
                raise ValueError("Missing candidate clip")
            records = {}
            for candidate in candidates:
                clip = by_id[candidate.case_id]
                if abs(clip.start - candidate.start) > 1e-6 or abs(clip.end - candidate.end) > 1e-6:
                    raise ValueError("Candidate bounds do not match")
                with wave.open(clip.path, "rb") as wav:
                    frames, rate = wav.getnframes(), wav.getframerate()
                    if (wav.getnchannels() != 1 or wav.getsampwidth() != 2 or rate != 16000
                            or abs(frames / rate - (candidate.end - candidate.start)) > 1 / rate + 1e-7):
                        raise ValueError("Candidate audio is truncated or has unexpected geometry")
                with Path(clip.path).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                case_id = candidate.case_id.removesuffix("-candidate")
                records[case_id] = {
                    "case_id": case_id, "candidate_audio_id": candidate.case_id,
                    "start": clip.start, "end": clip.end, "sha256": digest, "frames": frames,
                    "sample_rate": rate, "path": clip.path,
                    "policy_version": SOURCE_PAIR_CANDIDATE_AUDIO_POLICY_VERSION,
                }
        except (AudioSnippetError, OSError, ValueError, KeyError, wave.Error) as exc:
            raise ProviderError("Source pair candidate audio is missing, changed or incomplete.") from exc
        with self._lock:
            self._records.update(records)
        return method(spans, {**snippets, **by_id})
