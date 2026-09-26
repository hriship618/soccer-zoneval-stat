from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MatchSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def partition(self, match_id: str) -> str:
        if match_id in self.train:
            return "train"
        if match_id in self.validation:
            return "validation"
        if match_id in self.test:
            return "test"
        raise KeyError(match_id)


def chronological_split(
    matches: Iterable[tuple[str, str]],
    train_cutoff: str,
    validation_cutoff: str,
) -> MatchSplit:
    """Split complete matches by ISO date, never overlapping windows."""
    train_end = date.fromisoformat(train_cutoff)
    validation_end = date.fromisoformat(validation_cutoff)
    if validation_end <= train_end:
        raise ValueError("validation_cutoff must be after train_cutoff")
    groups = {"train": [], "validation": [], "test": []}
    for match_id, kickoff in sorted(matches, key=lambda row: row[1]):
        match_date = date.fromisoformat(kickoff[:10])
        group = "train" if match_date <= train_end else "validation" if match_date <= validation_end else "test"
        groups[group].append(match_id)
    return MatchSplit(*(tuple(groups[name]) for name in ("train", "validation", "test")))


def expanding_folds(match_ids: Sequence[str], minimum_train_matches: int = 3) -> list[MatchSplit]:
    if minimum_train_matches < 1:
        raise ValueError("minimum_train_matches must be positive")
    folds = []
    for boundary in range(minimum_train_matches, len(match_ids)):
        folds.append(MatchSplit(tuple(match_ids[:boundary]), (), (match_ids[boundary],)))
    return folds
