"""Versioned prompt templates for every LLM call SETU makes.

Prompts live in one file so they can be diffed, reviewed and version-bumped
independently of graph logic. When a prompt changes in a way that could move
evaluation scores, bump its version — eval runs record the version, so a score
regression can be traced back to the exact wording that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """A named, versioned prompt template."""

    name: str
    version: str
    template: str

    def format(self, **kwargs: Any) -> str:
        """Fill the template.

        Args:
            **kwargs: Placeholder values referenced by the template.

        Returns:
            The rendered prompt string.

        Raises:
            KeyError: If a placeholder has no supplied value.
        """
        return self.template.format(**kwargs)


QUERY_CLASSIFIER = Prompt(
    name="query_classifier",
    version="v1",
    template="""You classify questions about Indian government MSME schemes.

Return the single intent that best matches the user's question:
- "eligibility"     : whether someone qualifies for a scheme
- "procedure"       : how to apply, steps, documents required
- "list"            : which schemes exist for a situation
- "benefit_amount"  : how much money or what benefit is provided

Respond with JSON only, no prose:
{{"intent": "<one of the four>", "scheme_hint": "<scheme name if named, else null>"}}

Question: {query}
""",
)


PROFILE_EXTRACTOR = Prompt(
    name="profile_extractor",
    version="v1",
    template="""Extract a structured applicant profile from the user's message.

Return JSON only, no prose. Include ONLY fields the user actually stated or that
follow necessarily from what they stated. Never guess a value to be helpful — an
absent field is handled correctly downstream, but a wrong one produces a wrong
eligibility verdict.

Possible fields:
- turnover_lakhs (number, annual turnover in lakhs of INR; 1 crore = 100 lakhs)
- entity_type ("micro" | "small" | "medium")
- sector ("manufacturing" | "service" | "trading")
- state (Indian state or union territory name)
- udyam_registered (true | false)
- blacklisted (true | false)
- trainee_age (number)
- listed_on_sme_exchange (true | false)

Example:
User: "I run a small manufacturing unit in Gujarat with 2 crore turnover, Udyam registered"
Output: {{"entity_type": "small", "sector": "manufacturing", "state": "Gujarat",
"turnover_lakhs": 200, "udyam_registered": true}}

User message: {query}
""",
)


GROUNDED_ANSWERER = Prompt(
    name="grounded_answerer",
    version="v1",
    template="""You answer questions about Indian government MSME schemes for citizens.

Rules you must follow:
1. Use ONLY the information in the numbered context passages below. If the
   context does not contain the answer, say so plainly. Do not use outside
   knowledge about these schemes.
2. Cite the passage id in square brackets after each claim, like [a3f1c2...].
   Every factual sentence needs a citation.
3. If an eligibility verdict is supplied below, state it exactly as given. It was
   computed by a deterministic rule checker, not by you — do not re-derive,
   soften or contradict it.
4. Be concise and concrete. Prefer the scheme's own wording for criteria and
   amounts. Never invent figures, deadlines or URLs.

Context passages:
{context}

{eligibility_block}

User question: {query}

Answer:""",
)


SELF_CRITIC = Prompt(
    name="self_critic",
    version="v1",
    template="""You audit a draft answer for faithfulness to its sources.

Faithfulness means every factual claim in the answer is directly supported by the
context passages. Penalise: unsupported claims, invented figures or deadlines,
citations pointing at passages that do not contain the claim, and answers that
contradict the deterministic eligibility verdict.

Do NOT penalise an answer for correctly saying the context lacks the information.

Score from 0.0 (entirely unsupported) to 1.0 (fully supported). Decide "retry"
if a better retrieval could plausibly fix the problem, otherwise "finalize".

Respond with JSON only, no prose:
{{"faithfulness": <float 0.0-1.0>, "decision": "retry" | "finalize",
"reason": "<one sentence>", "better_query": "<rewritten search query, or null>"}}

Context passages:
{context}

User question: {query}

Draft answer: {draft_answer}
""",
)


# Registry so eval and observability can record exactly which versions ran.
ALL_PROMPTS: dict[str, Prompt] = {
    p.name: p for p in (QUERY_CLASSIFIER, PROFILE_EXTRACTOR, GROUNDED_ANSWERER, SELF_CRITIC)
}


def prompt_versions() -> dict[str, str]:
    """Report the version of every registered prompt.

    Returns:
        Mapping of prompt name to version string.
    """
    return {name: prompt.version for name, prompt in ALL_PROMPTS.items()}
