"""Clean raw lecture transcripts (timestamped cue format) into plain paragraph text
ready for chunking and embedding.

Usage:
    uv run python -m llmops.clean_raw [--src data/raw] [--dst data/clean-raw]
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

TIMESTAMP_RE = re.compile(
    r"^\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)

# Whole-cue utterances that carry no content (classroom fillers / check-ins).
FILLER_RE = re.compile(
    r"^(ok(ay)?|right|yes|yeah|yep|no|hmm+|um+|uh+|see|great|perfect|nice|good|"
    r"fine|clear|correct|thank you|thanks|"
    r"(ok(ay)?|so|right|yes)?\s*(guys)?\s*(is|was)?\s*(it|this|that|everything)?\s*clear( guys)?|"
    r"am i audible|can you see my screen|ok(ay)? guys|ok(ay)? fine|"
    r"any (other |more )?(questions?|doubts?)( guys| anyone)?)"
    r"[\s?.!,]*$",
    re.IGNORECASE,
)

PARAGRAPH_GAP_SECONDS = 6.0  # silence that starts a new paragraph
PARAGRAPH_MAX_WORDS = 180  # soft cap; break at the next cue boundary


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _norm_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    # collapse word loops like "NVIDIA NVIDIA NVIDIA NVIDIA" / "no no no" -> one occurrence
    text = re.sub(r"\b(\w+)(?:[\s,]+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    # collapse repeated phrase loops (2-6 words repeated 3+ times)
    text = re.sub(
        r"\b((?:\w+\W+){1,5}?\w+)(?:\W+\1\b){2,}", r"\1", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_cues(raw: str) -> list[tuple[float, float, str]]:
    """Return (start, end, text) cues; text lines without a timestamp attach to the previous cue."""
    cues: list[list] = []
    for line in raw.splitlines():
        m = TIMESTAMP_RE.match(line)
        if m:
            g = m.groups()
            cues.append([_secs(*g[:4]), _secs(*g[4:]), []])
        elif line.strip():
            if not cues:
                cues.append([0.0, 0.0, []])
            cues[-1][2].append(line.strip())
    return [(s, e, " ".join(t)) for s, e, t in cues if t]


def clean_transcript(raw: str) -> str:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    words = 0
    prev_end = None
    recent: list[str] = []  # keys of last few kept cues, for de-duplication

    for start, end, text in parse_cues(raw):
        text = _norm_line(text)
        if not text or FILLER_RE.match(text) or not re.search(r"[A-Za-z]", text):
            prev_end = end
            continue
        k = _key(text)
        if k in recent:  # repeated cue (common transcription artifact)
            prev_end = end
            continue
        recent = (recent + [k])[-3:]

        gap = start - prev_end if prev_end is not None else 0
        if current and (gap >= PARAGRAPH_GAP_SECONDS or words >= PARAGRAPH_MAX_WORDS):
            paragraphs.append(current)
            current, words = [], 0
        current.append(text)
        words += len(text.split())
        prev_end = end

    if current:
        paragraphs.append(current)

    out = []
    for p in paragraphs:
        parts = []
        for i, cue in enumerate(p):
            nxt = p[i + 1] if i + 1 < len(p) else ""
            # cue ends without punctuation and next cue starts a new sentence -> add a period
            if (not nxt or nxt[0].isupper()) and not re.search(r"[.!?,;:]$", cue):
                cue += "."
            parts.append(cue)
        para = " ".join(parts)
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            para = para[0].upper() + para[1:]
            out.append(para)
    return "\n".join(out) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=root / "data" / "raw")
    ap.add_argument("--dst", type=Path, default=root / "data" / "clean-raw")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.txt"), key=lambda p: (len(p.stem), p.stem))
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_transcript(raw)
        (args.dst / f.name).write_text(cleaned, encoding="utf-8", newline="\n")
        print(f"{f.name:>8}: {len(raw):>8,} -> {len(cleaned):>8,} chars, "
              f"{cleaned.count(chr(10)):>4} paragraphs")


if __name__ == "__main__":
    main()
