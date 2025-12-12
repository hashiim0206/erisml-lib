from __future__ import annotations

from dataclasses import dataclass
from typing import List

from erisml.ethics.facts import EthicalFacts
from erisml.ethics.judgement import EthicalJudgement, Verdict
from erisml.ethics.modules.base import EthicsModule


@dataclass
class CaseStudy1TriageEM(EthicsModule):
    """
    Example triage ethics module for Case Study 1.

    Uses only EthicalFacts, never raw ICD codes or clinical data. It:
      - Hard-forbids options that violate rights or explicit rules.
      - Otherwise computes a weighted composite score based on benefit,
        harm, urgency, disadvantaged priority, and procedural legitimacy.
    """

    em_name: str = "case_study_1_triage"
    stakeholder: str = "patients_and_public"

    # Weights over ethical dimensions (summing to ~1.0)
    w_benefit: float = 0.35
    w_harm: float = 0.25
    w_urgency: float = 0.20
    w_disadvantaged: float = 0.15
    w_procedural: float = 0.05

    def judge(self, facts: EthicalFacts) -> EthicalJudgement:
        # 1. Hard deontic veto: rights / explicit rule violations → forbid.
        rd = facts.rights_and_duties
        if rd.violates_rights or rd.violates_explicit_rule:
            reasons: List[str] = [
                (
                    "Option is forbidden because it violates fundamental rights "
                    "and/or explicit rules or regulations."
                )
            ]
            if rd.violates_rights:
                reasons.append("• violates_rights = True")
            if rd.violates_explicit_rule:
                reasons.append("• violates_explicit_rule = True")

            return EthicalJudgement(
                option_id=facts.option_id,
                em_name=self.em_name,
                stakeholder=self.stakeholder,
                verdict="forbid",
                normative_score=0.0,
                reasons=reasons,
                metadata={"kind": "hard_veto"},
            )

        # 2. Compute composite score from EthicalFacts.
        c = facts.consequences
        j = facts.justice_and_fairness
        p = facts.procedural_and_legitimacy

        # If procedural block missing, treat as neutral.
        procedural_score = 0.5
        if p is not None:
            procedural_score = 0.0
            if p.followed_approved_procedure:
                procedural_score += 0.5
            if p.stakeholders_consulted:
                procedural_score += 0.25
            if p.decision_explainable_to_public:
                procedural_score += 0.25

        benefit_term = c.expected_benefit
        harm_term = 1.0 - c.expected_harm  # lower harm → higher score
        urgency_term = c.urgency
        disadvantaged_term = 1.0 if j.prioritizes_most_disadvantaged else 0.0

        score = (
            self.w_benefit * benefit_term
            + self.w_harm * harm_term
            + self.w_urgency * urgency_term
            + self.w_disadvantaged * disadvantaged_term
            + self.w_procedural * procedural_score
        )

        # 3. Map score → verdict.
        if score >= 0.8:
            verdict: Verdict = "strongly_prefer"
        elif score >= 0.6:
            verdict = "prefer"
        elif score >= 0.4:
            verdict = "neutral"
        elif score >= 0.2:
            verdict = "avoid"
        else:
            verdict = "forbid"

        reasons: List[str] = [
            (
                "Composite triage judgement based on benefit, harm, urgency, "
                "priority for the disadvantaged, autonomy, and procedural legitimacy."
            )
        ]

        return EthicalJudgement(
            option_id=facts.option_id,
            em_name=self.em_name,
            stakeholder=self.stakeholder,
            verdict=verdict,
            normative_score=score,
            reasons=reasons,
            metadata={"kind": "triage_em"},
        )


@dataclass
class RightsFirstEM(EthicsModule):
    """
    Simple rights-compliance EM.

    - Forbids any option that violates rights or explicit rules.
    - Otherwise returns a fixed 'prefer' with a high normative score.
    This is intended to plug into governance as a veto-capable EM.
    """

    em_name: str = "rights_first_compliance"
    stakeholder: str = "patients_and_public"

    def judge(self, facts: EthicalFacts) -> EthicalJudgement:
        rd = facts.rights_and_duties
        reasons: List[str] = []
        verdict: Verdict
        score: float

        if rd.violates_rights or rd.violates_explicit_rule:
            verdict = "forbid"
            score = 0.0
            reasons.append(
                "Forbid: option violates rights and/or explicit rules, "
                "which take precedence over other considerations."
            )
            if rd.violates_rights:
                reasons.append("• violates_rights = True")
            if rd.violates_explicit_rule:
                reasons.append("• violates_explicit_rule = True")
        else:
            verdict = "prefer"
            score = 0.8
            reasons.append(
                "Rights and explicit rules are respected; "
                "no deontic veto from this module."
            )

        return EthicalJudgement(
            option_id=facts.option_id,
            em_name=self.em_name,
            stakeholder=self.stakeholder,
            verdict=verdict,
            normative_score=score,
            reasons=reasons,
            metadata={"kind": "rights_first"},
        )


__all__ = ["CaseStudy1TriageEM", "RightsFirstEM"]
