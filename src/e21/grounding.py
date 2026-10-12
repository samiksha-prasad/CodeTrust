"""The mitigation: look the code up in the official table before the model answers.

The lookup result is handed to the model as a system prompt, so the model keeps doing the
explaining while the table decides whether there is anything to explain.
"""

import re

from .schema import CodeEntry

# Uppercase tokens that look like codes but are not.
_NOT_CODES = {"AC", "OK", "TV", "UPI", "OBD", "II", "ATM", "PIN", "OTP"}

_SAFE_STEPS = (
    "Offer only general safe steps (switch off and unplug before touching anything, do not open the "
    "casing, do not repeat a payment until the first one's status is known) and point them to the "
    "manufacturer's or provider's official support."
)


class Grounder:
    def __init__(self, entries: list[CodeEntry], complete: set[tuple[str, str]]):
        self.entries = entries
        self.complete = complete
        self.brands = sorted({e.brand for e in entries if e.brand} | {b for _, b in complete if b})
        self.devices = sorted({e.device for e in entries})
        self.known_codes = {e.code.upper() for e in entries}

    def _candidates(self, text: str) -> list[str]:
        skip = {b.upper() for b in self.brands} | _NOT_CODES
        tokens = [t for t in re.findall(r"[A-Za-z0-9]+", text) if t.upper() not in skip]
        known = [t for t in tokens if t.upper() in self.known_codes]

        def code_like(t: str) -> bool:
            if not 2 <= len(t) <= 6 or t.isdigit():
                return False
            return any(c.isdigit() for c in t) or not (t.islower() or t.istitle())

        return known + [t for t in tokens if code_like(t) and t not in known]

    def lookup(self, text: str) -> dict | None:
        candidates = self._candidates(text)
        if not candidates:
            return None
        code = candidates[0]
        brands = [b for b in self.brands if re.search(rf"\b{re.escape(b)}\b", text, re.IGNORECASE)]
        devices = [d for d in self.devices if d.lower() in text.lower()] or self.devices
        matches = [e for e in self.entries if e.device in devices and e.code.upper() == code.upper()]

        if brands:
            brand = brands[0]
            hit = next((e for e in matches if e.brand == brand), None)
            if hit:
                return {"status": "found", "code": code, "entry": hit}
            if any((d, brand) in self.complete for d in devices):
                return {"status": "not_in_brand_table", "code": code, "brand": brand}
            return {"status": "unknown_code", "code": code}

        standard = next((e for e in matches if not e.brand), None)
        if standard:
            return {"status": "found", "code": code, "entry": standard}
        if matches:
            return {"status": "need_brand", "code": code, "brands": sorted({e.brand for e in matches})}
        return {"status": "unknown_code", "code": code}

    def system_prompt(self, text: str) -> str | None:
        hit = self.lookup(text)
        if hit is None:
            return None
        code = hit["code"]
        if hit["status"] == "found":
            e = hit["entry"]
            name = f"{e.brand} {e.device}".strip()
            return (
                f"The user is asking about an error code. The official documentation for {name} lists "
                f"code {e.code} as: {e.meaning}. Officially recommended action: {e.official_action}. "
                f"Source: {e.source_url or 'official manual'}. Base your explanation on this entry. Any "
                "extra advice must be consistent with it, and should not go beyond what an ordinary user "
                "can safely do."
            )
        if hit["status"] == "not_in_brand_table":
            return (
                f"The user is asking about error code {code} on a {hit['brand']} device. That code is not "
                f"in the official {hit['brand']} error-code list, and that list is complete. Tell them "
                "plainly that you cannot find this code for their brand, and do not offer a meaning for "
                "it: a guessed meaning can send them to the wrong repair. Suggest they re-read the "
                f"display in case characters were misread. {_SAFE_STEPS}"
            )
        if hit["status"] == "need_brand":
            return (
                f"The user is asking about error code {code} without naming a brand. This code is used by "
                f"more than one brand ({', '.join(hit['brands'])}) and may mean different things on each. "
                "Ask which brand and model they have before explaining what the code means."
            )
        return (
            f"The user is asking about error code {code}. There is no verified official entry for this "
            "code. Say that you cannot confirm what it means, ask for the exact brand and model, and do "
            f"not guess a meaning. {_SAFE_STEPS}"
        )


class GroundedModel:
    def __init__(self, base, grounder: Grounder):
        self.base = base
        self.grounder = grounder
        self.name = f"grounded+{base.name}"

    def ask(self, prompt: str, system: str | None = None) -> str:
        return self.base.ask(prompt, system=self.grounder.system_prompt(prompt))
