"""Signs that text was written by an AI chat assistant (ChatGPT, Gemini, Claude…), shown as evidence, not a verdict.

No detector can prove who wrote a text, and AI-text classifiers often misfire on formal or non-native writing, so
TruthLens only reports signs a reader can check in the text itself:

- Chatbot leftovers: replies pasted with the assistant's own words ("Certainly! Here's…", "As an AI language model"),
  unfilled template placeholders ("[Your Name]") and leftover Markdown (**bold**, ## headings). Close to direct evidence.
- Over-used vocabulary: words and phrases whose frequency jumped after ChatGPT appeared (e.g. "delve", "pivotal",
  "showcasing", "a testament to"; Kobak et al. 2024, "Delving into ChatGPT usage in academic writing through excess
  vocabulary"). Measured per 1,000 words and listed with counts. Weak on its own: formal reports and grammar tools
  use the same words.
"""
from __future__ import annotations

import re

from .config import RELIABILITY as R

LEFTOVERS = [
    (r"\bas an AI(?: language)? model\b", "“as an AI language model”"),
    (r"(?im)^\s*(?:certainly|sure|absolutely|of course)[!,.]?\s+here(?:'s|’s| is| are)\b", "“Certainly! Here's…” opening"),
    (r"\bI hope (?:this|that) helps\b", "“I hope this helps”"),
    (r"\b(?:let me know|feel free to ask) if you(?:'d|’d| would)? (?:like|want|need)\b", "“Let me know if you'd like…”"),
    (r"\[(?:your|insert|add|enter|company|recipient|date|name)[^\]\n]{0,40}\]", "unfilled placeholder like “[Your Name]”"),
    (r"\*\*[^*\n]{2,80}\*\*", "leftover Markdown bold (**…**)"),
    (r"(?m)^#{1,4} \S", "leftover Markdown heading (## …)"),
    (r"\b(?:knowledge cutoff|my training data)\b", "talks about its own training"),
]
VOCAB = [
    r"delv(?:e|es|ed|ing)", r"showcas(?:e|es|ed|ing)", r"underscor(?:e|es|ed|ing)", r"pivotal", r"intricate",
    r"meticulous(?:ly)?", r"commendable", r"realm", r"tapestry", r"multifaceted", r"nuanced", r"paramount",
    r"invaluable", r"noteworthy", r"groundbreaking", r"transformative", r"seamless(?:ly)?", r"robust", r"holistic",
    r"foster(?:s|ed|ing)?", r"leverag(?:e|es|ed|ing)", r"synerg(?:y|ies|istic)", r"streamlin(?:e|es|ed|ing)",
    r"unwavering", r"bustling", r"vibrant", r"ever-evolving", r"cutting-edge", r"interplay", r"endeavou?r(?:s)?",
    r"embark(?:s|ed|ing)?", r"elevat(?:e|es|ed|ing)", r"navigat(?:e|es|ed|ing) the", r"landscape", r"ecosystem",
    r"pragmatism", r"testament", r"resonat(?:e|es|ed|ing)", r"profound(?:ly)?", r"crucial", r"comprehensive",
    r"enhanc(?:e|es|ed|ing)", r"insightful", r"(?:valuable|practical|deeper) insights?",
    r"plays? an? (?:crucial|pivotal|vital|key|significant) role", r"bridg(?:e|es|ed|ing) the gap",
    r"in today's (?:fast-paced|digital|rapidly)", r"it is (?:important|worth|crucial) to note",
    r"not only\b[^.]{0,60}\bbut also", r"serv(?:e|es|ed) as a", r"in conclusion", r"overall,",
]
VOCAB_RE = re.compile(r"\b(?:" + "|".join(VOCAB) + r")\b", re.I)
MIN_WORDS = 150
RATE_START, RATE_FULL = 4.0, 16.0   # hits per 1,000 words: no signal below 4, full signal at 16


def check_ai_writing(out, text: str) -> dict:
    """Adds `text_chatbot_leftover` / `text_ai_vocabulary` indicators to `out` (a Findings) and returns metrics."""
    words = len(text.split())
    m = {"words": words, "ai_vocab_per_1000": None, "ai_vocab_top": [], "chatbot_leftovers": []}
    if words < MIN_WORDS:
        m["skipped"] = f"only {words} words; at least {MIN_WORDS} are needed to judge writing style"
        return m
    found = []
    for pat, label in LEFTOVERS:
        hits = re.findall(pat, text, re.I)
        if hits:
            found.append((label, len(hits), (hits[0] if isinstance(hits[0], str) else "")[:60]))
    m["chatbot_leftovers"] = [f"{label} ×{n}" for label, n, _ in found]
    if found:
        strong = [f for f in found if "Markdown" not in f[0]]
        out.add("text_chatbot_leftover", "Text contains chatbot leftovers", "synthetic",
                "Found: " + "; ".join(f"{label} ({n}×, e.g. “{ex}”)" if ex else f"{label} ({n}×)" for label, n, ex in found) + ".",
                "These are an AI assistant's own phrases or formatting, left in when its reply was pasted into the text.",
                0.95 if strong else 0.6, R["text_chatbot_leftover"], {"type": "file"},
                ["A person quoting or discussing a chatbot reply.", "Markdown can also come from note apps such as Notion or Obsidian."])
    counts: dict[str, int] = {}
    for h in VOCAB_RE.findall(text):
        k = h.lower()
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    rate = 1000 * total / words
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    m["ai_vocab_per_1000"], m["ai_vocab_top"] = round(rate, 1), [f"{w} ×{n}" for w, n in top]
    s = min(1.0, max(0.0, (rate - RATE_START) / (RATE_FULL - RATE_START)))
    if s > 0:
        out.add("text_ai_vocabulary", "Writing style typical of AI chat assistants", "synthetic",
                f"{total} uses of words and phrases that AI assistants over-use, {rate:.1f} per 1,000 words "
                f"(we start counting it as a sign above {RATE_START:.0f}): " + ", ".join(f"“{w}” ×{n}" for w, n in top) + ".",
                "The frequency of these words jumped in published writing after ChatGPT appeared, because AI assistants "
                "use them far more than people do.",
                s, R["text_ai_vocabulary"], {"type": "file"},
                ["Formal reports, essays and business writing use some of these words.",
                 "Grammar and paraphrasing tools (Grammarly, QuillBot) suggest them.",
                 "A person may have written the text and asked AI only to polish it."])
    return m
