"""Natural-interrupt mouth-closing feature: openness scoring, a reusable
closing-animation library captured from natural end-of-speech transitions,
selection at barge-in time, and a dedicated log for the whole flow.

Pure OpenCV/numpy -- no extra ML dependency, so it drops into the existing
pinned, checksum-verified deployment without touching the install script.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BUCKET_ORDER = ["closed", "quarter", "half", "three_quarter", "full"]


def _bucket_thresholds() -> list[float]:
    raw = os.environ.get("IMTALKER_MOUTH_BUCKET_THRESHOLDS", "0.06,0.16,0.30,0.48")
    try:
        vals = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        vals = []
    if len(vals) != 4:
        vals = [0.06, 0.16, 0.30, 0.48]
    return vals


_THRESHOLDS = _bucket_thresholds()
CLOSED_SCORE_THRESHOLD = float(os.environ.get("IMTALKER_MOUTH_CLOSED_SCORE", "0.045"))


def bucket_for_score(score: float) -> str:
    for i, t in enumerate(_THRESHOLDS):
        if score < t:
            return BUCKET_ORDER[i]
    return BUCKET_ORDER[-1]


class MouthOpennessScorer:
    """Heuristic mouth-openness scorer using only the OpenCV Haar cascade
    already bundled with opencv-python (no extra model/dependency).

    Approach: detect the face once (cached, periodically re-detected to
    tolerate small framing drift), crop a fixed-proportion mouth region
    inside the face box, and score openness as the height of the darkest
    connected blob near the ROI's vertical center, normalized by ROI
    height. This is a coarse proxy tuned for bucketing a fixed, frontal
    avatar into a handful of "how open is the mouth" tiers -- not a
    precise landmark measurement. Thresholds are tunable via env vars;
    verify visually against the saved frame dumps and adjust if needed.
    """

    def __init__(self, redetect_every_s: float = 5.0) -> None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        self._redetect_every_s = redetect_every_s
        self._lock = threading.Lock()
        self._face_bbox: Optional[tuple[int, int, int, int]] = None
        self._last_detect_wall = 0.0

    def _detect_face(self, gray: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        min_dim = max(32, gray.shape[1] // 4)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_dim, min_dim)
        )
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))

    def score_bgr(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Return an openness score in [0, 1], or None if no face is found."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        now = time.perf_counter()
        with self._lock:
            bbox = self._face_bbox
            need_redetect = (
                bbox is None or (now - self._last_detect_wall) > self._redetect_every_s
            )
        if need_redetect:
            detected = self._detect_face(gray)
            if detected is not None:
                bbox = detected
                with self._lock:
                    self._face_bbox = bbox
                    self._last_detect_wall = now
        if bbox is None:
            return None

        x, y, w, h = bbox
        mx0 = max(0, x + int(0.28 * w))
        mx1 = min(gray.shape[1], x + int(0.72 * w))
        my0 = max(0, y + int(0.62 * h))
        my1 = min(gray.shape[0], y + int(0.94 * h))
        if mx1 <= mx0 or my1 <= my0:
            return None
        roi = gray[my0:my1, mx0:mx1]
        if roi.size == 0:
            return None

        _thr, mask = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        num, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num <= 1:
            return 0.0

        roi_h, roi_w = roi.shape
        cx_target = roi_w / 2.0
        best_area = 0
        best_idx: Optional[int] = None
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            cx = float(centroids[i][0])
            if abs(cx - cx_target) > roi_w * 0.4:
                continue
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx is None:
            return 0.0

        blob_h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        return float(np.clip(blob_h / max(1.0, roi_h), 0.0, 1.0))

    def score_jpeg_bytes(self, jpeg_bytes: bytes) -> Optional[float]:
        if not jpeg_bytes:
            return None
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None
        return self.score_bgr(frame_bgr)


class InterruptCloseLogger:
    """Dedicated logger for the interrupt/closing-animation subsystem.

    Writes to its own file (never mixed into the general server stdout
    prints) plus echoes to the console so it's visible in the live tail too.
    """

    def __init__(self, log_path: str) -> None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("imtalker.interrupt_closing")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s.%(msecs)03d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
            self._logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("[INTERRUPT-CLOSE] %(message)s"))
            self._logger.addHandler(console_handler)
        self.log_path = log_path

    def _emit(self, event: str, **fields) -> None:
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        self._logger.info("%s %s", event, payload)

    def capture_created(self, clip_id, bucket, n_frames, source_chunk_id, saved_dir, duration_ms) -> None:
        self._emit(
            "CAPTURE_CREATED", clip_id=clip_id, bucket=bucket, frames=n_frames,
            source_chunk_id=source_chunk_id, saved_dir=saved_dir, duration_ms=f"{duration_ms:.1f}",
        )

    def capture_skipped(self, reason, source_chunk_id=None) -> None:
        self._emit("CAPTURE_SKIPPED", reason=reason, source_chunk_id=source_chunk_id)

    def capture_error(self, source_chunk_id, exc) -> None:
        self._emit("CAPTURE_ERROR", source_chunk_id=source_chunk_id, exc=repr(exc))

    def interrupt_detected(self, session_id, generation) -> None:
        self._emit("INTERRUPT_DETECTED", session=session_id[:8], generation=generation)

    def mouth_measured(self, session_id, score, bucket, measured_from) -> None:
        score_s = f"{score:.3f}" if score is not None else "None"
        self._emit(
            "MOUTH_MEASURED", session=session_id[:8], score=score_s, bucket=bucket, source=measured_from
        )

    def clip_selected(self, session_id, clip_id, bucket, n_frames, reason) -> None:
        self._emit(
            "CLIP_SELECTED", session=session_id[:8], clip_id=clip_id, bucket=bucket,
            frames=n_frames, reason=reason,
        )

    def playback_start(self, session_id, clip_id, n_frames) -> None:
        self._emit("CLOSING_PLAYBACK_START", session=session_id[:8], clip_id=clip_id, frames=n_frames)

    def playback_end(self, session_id, clip_id, success, duration_ms, frames_sent) -> None:
        self._emit(
            "CLOSING_PLAYBACK_END", session=session_id[:8], clip_id=clip_id, success=success,
            duration_ms=f"{duration_ms:.1f}", frames_sent=frames_sent,
        )

    def idle_video_start(self, session_id, reason) -> None:
        self._emit("IDLE_VIDEO_START", session=session_id[:8], reason=reason)

    def fallback(self, session_id, reason) -> None:
        self._emit("FALLBACK", session=session_id[:8], reason=reason)

    def error(self, session_id, where, exc) -> None:
        self._emit("ERROR", session=(session_id[:8] if session_id else "-"), where=where, exc=repr(exc))


@dataclass
class ClosingClip:
    clip_id: str
    bucket: str
    frame_scores: list
    jpeg_frames: list
    created_at: float
    source_chunk_id: int
    saved_dir: Optional[str] = None


class ClosingAnimationLibrary:
    """In-memory store of reusable closing clips, bucketed by starting
    mouth-openness, backed by an on-disk frame dump per clip for manual
    inspection."""

    def __init__(self, save_root: str, max_clips_per_bucket: int = 8) -> None:
        self._save_root = Path(save_root)
        self._save_root.mkdir(parents=True, exist_ok=True)
        self._max_per_bucket = max_clips_per_bucket
        self._lock = threading.Lock()
        self._clips: dict[str, list[ClosingClip]] = {b: [] for b in BUCKET_ORDER}
        self._rr_index: dict[str, int] = {b: 0 for b in BUCKET_ORDER}

    def save_root_for(self, bucket: str, clip_id: str) -> Path:
        d = self._save_root / bucket / clip_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_clip(self, clip: ClosingClip) -> None:
        with self._lock:
            bucket_list = self._clips[clip.bucket]
            bucket_list.append(clip)
            if len(bucket_list) > self._max_per_bucket:
                bucket_list.pop(0)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {b: len(v) for b, v in self._clips.items()}

    def pick_clip(self, target_bucket: str) -> tuple[Optional[ClosingClip], str]:
        with self._lock:
            ti = BUCKET_ORDER.index(target_bucket)
            order = sorted(BUCKET_ORDER, key=lambda b: abs(BUCKET_ORDER.index(b) - ti))
            for b in order:
                bucket_list = self._clips[b]
                if bucket_list:
                    idx = self._rr_index[b] % len(bucket_list)
                    self._rr_index[b] += 1
                    reason = "exact_bucket" if b == target_bucket else f"nearest_bucket({b})"
                    return bucket_list[idx], reason
            return None, "library_empty"


def _save_clip_frames(out_dir: Path, jpeg_frames: list, scores: list) -> Path:
    manifest = []
    for i, (jb, sc) in enumerate(zip(jpeg_frames, scores)):
        fname = f"frame_{i:03d}.jpg"
        with open(out_dir / fname, "wb") as f:
            f.write(jb)
        manifest.append({"frame": i, "file": fname, "openness_score": round(float(sc), 4)})
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"frame_count": len(jpeg_frames), "frames": manifest}, f, indent=2)
    return out_dir


def finalize_capture_clip(
    jpeg_frames: list,
    scorer: MouthOpennessScorer,
    library: ClosingAnimationLibrary,
    logger: InterruptCloseLogger,
    source_chunk_id: int,
) -> None:
    """Runs off the hot path (submit to a background executor). Scores each
    captured frame, trims the clip shortly after the mouth reaches the
    "closed" threshold, and registers it in the library under the bucket of
    its *starting* openness."""
    t0 = time.perf_counter()
    try:
        if not jpeg_frames:
            logger.capture_skipped("empty_batch", source_chunk_id)
            return

        scores = []
        for jb in jpeg_frames:
            s = scorer.score_jpeg_bytes(jb)
            scores.append(s if s is not None else (scores[-1] if scores else 1.0))

        first_score = scores[0]
        if first_score < CLOSED_SCORE_THRESHOLD:
            logger.capture_skipped("already_closed_at_transition", source_chunk_id)
            return

        end_idx = len(jpeg_frames)
        for i, s in enumerate(scores):
            if s < CLOSED_SCORE_THRESHOLD:
                end_idx = min(len(jpeg_frames), i + 2)
                break

        trimmed_frames = jpeg_frames[:end_idx]
        trimmed_scores = scores[:end_idx]
        if len(trimmed_frames) < 2:
            logger.capture_skipped("too_short", source_chunk_id)
            return

        bucket = bucket_for_score(first_score)
        clip_id = f"{bucket}-{uuid.uuid4().hex[:8]}"
        saved_dir = _save_clip_frames(
            library.save_root_for(bucket, clip_id), trimmed_frames, trimmed_scores
        )
        clip = ClosingClip(
            clip_id=clip_id,
            bucket=bucket,
            frame_scores=trimmed_scores,
            jpeg_frames=trimmed_frames,
            created_at=time.time(),
            source_chunk_id=source_chunk_id,
            saved_dir=str(saved_dir),
        )
        library.add_clip(clip)
        logger.capture_created(
            clip_id, bucket, len(trimmed_frames), source_chunk_id, str(saved_dir),
            (time.perf_counter() - t0) * 1000.0,
        )
    except Exception as exc:  # noqa: BLE001 - capture is best-effort, must never crash the GPU thread's caller
        logger.capture_error(source_chunk_id, exc)


def select_closing_clip(
    last_sent_jpeg: Optional[bytes],
    scorer: MouthOpennessScorer,
    library: ClosingAnimationLibrary,
    logger: InterruptCloseLogger,
    session_id: str,
) -> tuple[Optional[ClosingClip], Optional[float], str, str]:
    """Returns (clip, score, bucket, selection_reason)."""
    score: Optional[float] = None
    measured_from = "unavailable"
    if last_sent_jpeg:
        try:
            score = scorer.score_jpeg_bytes(last_sent_jpeg)
            measured_from = "last_sent_frame" if score is not None else "face_not_detected"
        except Exception as exc:  # noqa: BLE001
            logger.error(session_id, "score_jpeg_bytes", exc)
            measured_from = "score_error"
    bucket = bucket_for_score(score) if score is not None else "half"
    logger.mouth_measured(session_id, score, bucket, measured_from)
    clip, reason = library.pick_clip(bucket)
    return clip, score, bucket, reason
