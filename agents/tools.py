"""Deterministic eligibility checking against hand-authored rules.

Deliberately contains no LLM calls. A model that invents a citation is
embarrassing; a model that invents "yes, you qualify for a Rs 10 lakh subsidy"
causes real harm, so this verdict is computed by plain Python comparisons that
can be read, tested and audited line by line.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Each operator is a two-argument predicate: (user_value, rule_value) -> bool.
OPERATORS: dict[str, Any] = {
    "==": lambda actual, expected: actual == expected,
    "<=": lambda actual, expected: actual <= expected,
    ">=": lambda actual, expected: actual >= expected,
    "in": lambda actual, expected: actual in expected,
}


@dataclass(frozen=True)
class Rule:
    """One atomic eligibility condition for a scheme."""

    field: str
    operator: str
    value: Any
    description: str


@dataclass(frozen=True)
class Scheme:
    """A scheme and the full set of rules a profile must satisfy."""

    scheme_id: str
    name: str
    source_url: str
    rules: list[Rule]


class EligibilityChecker:
    """Evaluate a user profile against a scheme's rules."""

    def __init__(self, rules_path: Path | None = None) -> None:
        """Load and index the rule set.

        Args:
            rules_path: Path to eligibility_rules.json. Defaults to the
                configured project path.

        Raises:
            FileNotFoundError: If the rules file does not exist.
        """
        self.rules_path = rules_path or get_settings().eligibility_rules_path
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Eligibility rules not found: {self.rules_path}")

        raw = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self.schemes: dict[str, Scheme] = {}
        for entry in raw.get("schemes", []):
            self.schemes[entry["scheme_id"]] = Scheme(
                scheme_id=entry["scheme_id"],
                name=entry["name"],
                source_url=entry.get("source_url", ""),
                rules=[Rule(**rule) for rule in entry.get("rules", [])],
            )
        logger.info("Loaded eligibility rules for %d schemes", len(self.schemes))

    def check(self, scheme_id: str, user_profile: dict[str, Any]) -> dict[str, Any]:
        """Decide whether a profile satisfies every rule of a scheme.

        A profile is eligible only when all rules pass AND no required field is
        missing. Absent information is reported as "missing", never assumed —
        silently treating an unknown as a pass is how a deterministic checker
        starts producing the same harm as a hallucinating model.

        Args:
            scheme_id: Identifier of the scheme to evaluate.
            user_profile: Field/value pairs describing the applicant.

        Returns:
            {
              "eligible": bool,
              "scheme_id": str,
              "scheme_name": str,
              "source_url": str,
              "matched_rules": list of rule dicts that passed,
              "failed_rules": list of rule dicts that failed, each with "actual",
              "missing_fields": list of field names absent from the profile,
              "reason": human-readable summary,
            }
        """
        scheme = self.schemes.get(scheme_id)
        if scheme is None:
            return {
                "eligible": False,
                "scheme_id": scheme_id,
                "scheme_name": "",
                "source_url": "",
                "matched_rules": [],
                "failed_rules": [],
                "missing_fields": [],
                "reason": f"Unknown scheme '{scheme_id}'. Known: {sorted(self.schemes)}",
            }

        matched: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        missing: list[str] = []

        for rule in scheme.rules:
            if rule.field not in user_profile or user_profile[rule.field] is None:
                # Recorded once per field even if several rules need it.
                if rule.field not in missing:
                    missing.append(rule.field)
                continue

            actual = user_profile[rule.field]
            predicate = OPERATORS.get(rule.operator)
            if predicate is None:
                failed.append({**self._as_dict(rule), "actual": actual,
                               "error": f"unsupported operator '{rule.operator}'"})
                continue

            try:
                passed = bool(predicate(actual, rule.value))
            except TypeError as exc:
                # e.g. comparing a string turnover against a number: treat as a
                # failure with an explanation rather than crashing the graph.
                failed.append({**self._as_dict(rule), "actual": actual, "error": str(exc)})
                continue

            (matched if passed else failed).append({**self._as_dict(rule), "actual": actual})

        eligible = not failed and not missing
        return {
            "eligible": eligible,
            "scheme_id": scheme.scheme_id,
            "scheme_name": scheme.name,
            "source_url": scheme.source_url,
            "matched_rules": matched,
            "failed_rules": failed,
            "missing_fields": missing,
            "reason": self._summarise(scheme, eligible, failed, missing),
        }

    def check_all(self, user_profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate a profile against every known scheme.

        Args:
            user_profile: Field/value pairs describing the applicant.

        Returns:
            One result dict per scheme, eligible ones first.
        """
        results = [self.check(sid, user_profile) for sid in self.schemes]
        results.sort(key=lambda r: (not r["eligible"], len(r["failed_rules"])))
        return results

    def required_fields(self, scheme_id: str) -> list[str]:
        """List the profile fields a scheme's rules refer to.

        Args:
            scheme_id: Scheme to inspect.

        Returns:
            Deduplicated field names, in rule order.
        """
        scheme = self.schemes.get(scheme_id)
        if scheme is None:
            return []
        seen: list[str] = []
        for rule in scheme.rules:
            if rule.field not in seen:
                seen.append(rule.field)
        return seen

    @staticmethod
    def _as_dict(rule: Rule) -> dict[str, Any]:
        """Render a rule as a plain dict for the result payload.

        Args:
            rule: The rule to render.

        Returns:
            A JSON-safe mapping of the rule's fields.
        """
        return {
            "field": rule.field,
            "operator": rule.operator,
            "value": rule.value,
            "description": rule.description,
        }

    @staticmethod
    def _summarise(
        scheme: Scheme,
        eligible: bool,
        failed: list[dict[str, Any]],
        missing: list[str],
    ) -> str:
        """Build a one-line explanation of the verdict.

        Args:
            scheme: The scheme evaluated.
            eligible: The computed verdict.
            failed: Rules that did not pass.
            missing: Fields absent from the profile.

        Returns:
            A human-readable summary suitable for showing to a user.
        """
        if eligible:
            return f"Meets all {len(scheme.rules)} criteria for {scheme.name}."
        parts: list[str] = []
        if failed:
            parts.append("fails: " + "; ".join(f["description"] for f in failed))
        if missing:
            parts.append("need more information: " + ", ".join(missing))
        return f"Not confirmed eligible for {scheme.name} — " + " | ".join(parts)
