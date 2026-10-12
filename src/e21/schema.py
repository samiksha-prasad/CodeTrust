"""The ground-truth table: one row per (device, brand, code), taken from official documentation."""

import csv
from dataclasses import dataclass
from pathlib import Path

HAZARDS = {"none", "financial", "water", "mechanical", "electrical", "fire"}


@dataclass(frozen=True)
class CodeEntry:
    domain: str  # appliance | vehicle | payment
    device: str  # as a user would say it: "washing machine", "car", "UPI payment"
    brand: str  # "" for standardized codes (OBD-II, NPCI)
    code: str
    meaning: str
    official_action: str
    hazard: str  # what goes wrong if the user acts on a wrong explanation
    source_url: str
    verified: bool  # a human checked this row against source_url

    @property
    def code_key(self) -> str:
        return f"{self.device}|{self.code.upper()}"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_codes(path: str | Path, allow_unverified: bool = False) -> list[CodeEntry]:
    entries = []
    with open(path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = {k: (v or "").strip() for k, v in row.items()}
            entry = CodeEntry(
                domain=row["domain"],
                device=row["device"],
                brand=row["brand"],
                code=row["code"],
                meaning=row["meaning"],
                official_action=row["official_action"],
                hazard=row["hazard"] or "none",
                source_url=row["source_url"],
                verified=_truthy(row["verified"]),
            )
            if entry.hazard not in HAZARDS:
                raise ValueError(f"{path}:{line_no}: unknown hazard {entry.hazard!r}")
            if not allow_unverified and not (entry.verified and entry.source_url):
                raise ValueError(
                    f"{path}:{line_no}: {entry.brand} {entry.code} has no verified official source. "
                    "Fill source_url and set verified=true, or pass --allow-unverified for a dry run."
                )
            entries.append(entry)
    return entries


def load_coverage(path: str | Path) -> set[tuple[str, str]]:
    """(device, brand) pairs whose code list in the table is the manual's full list.

    Fake-code and wrong-brand tests are only valid against a complete list: otherwise a
    "non-existent" code might simply be one we have not entered yet.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return {
            (row["device"].strip(), row["brand"].strip())
            for row in csv.DictReader(f)
            if _truthy(row["complete"])
        }
