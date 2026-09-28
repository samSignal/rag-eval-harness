"""Text normalisation shared by metrics and the heuristic judge."""
from __future__ import annotations

import re

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100,
    "twice": 2, "double": 2,
}
STOP = set("a an the of to and or in on at for by with is are be was were it this that as from any all its "
           "their there these those than then shall may must can not no".split())
CITATION = re.compile(r"\[([A-Za-z0-9_.:#/-]+)\]")
REFUSAL = re.compile(r"(couldn't|could not|cannot|can't|unable to) (find|answer)|not (found |covered )?in the "
                     r"(provided |uploaded )?(documents|sources|context)|no information", re.I)


def normalise(text: str) -> str:
    """Lower-case, number words -> digits, '1,000' -> '1000', drop citation markers and punctuation."""
    text = CITATION.sub(" ", text.lower())
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: str(_NUM_WORDS[m.group(1)]), text)
    text = text.replace("%", " percent")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def terms(text: str) -> set[str]:
    return {t[:6] for t in normalise(text).split() if t not in STOP}  # prefix-stem: "notified" ~ "notify"


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", normalise(text)))


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", CITATION.sub("", text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer))
