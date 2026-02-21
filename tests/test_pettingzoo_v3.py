from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from erisml.ethics.coalition import CoalitionContext
from erisml.ethics.facts_v3 import (
    ConsequencesV3,
    EthicalFactsV3,
    JusticeAndFairnessV3,
    RightsAndDutiesV3,
)
from erisml.ethics.layers.strategic import (
    CoalitionStabilityAnalysis,
    StrategicAnalysisResult,
)
from erisml.ethics.moral_tensor import MoralTensor
from erisml.ethics.moral_vector import MoralVector
from erisml.interop.pettingzoo_adapter import ErisPettingZooEnv


class DummyModel:
    def __init__(self):
        self.agents = {"agent_0": object(), "agent_1": object()}
        self.env = SimpleNamespace(object_types={})


def _dummy_facts(_: dict) -> EthicalFactsV3:
    return EthicalFactsV3(
        option_id="opt",
        consequences=ConsequencesV3(
            expected_benefit=0.8, expected_harm=0.1, urgency=0.5, affected_count=2
        ),
        rights_and_duties=RightsAndDutiesV3(
            violates_rights=False,
            has_valid_consent=True,
            violates_explicit_rule=False,
            role_duty_conflict=False,
        ),
        justice_and_fairness=JusticeAndFairnessV3(
            discriminates_on_protected_attr=False,
            prioritizes_most_disadvantaged=True,
        ),
    )


def _build_judgement() -> MagicMock:
    judgement = MagicMock()
    judgement.moral_tensor = MoralTensor.zeros(shape=(9, 2))
    judgement.moral_tensor.axis_labels["n"] = ["agent_0", "agent_1"]
    judgement.get_party_vector.side_effect = lambda a: (
        MoralVector(rights_respect=1.0, physical_harm=0.0)
        if a == "agent_0"
        else MoralVector(rights_respect=0.0, physical_harm=1.0)
    )
    judgement.get_party_verdict.side_effect = lambda _: "neutral"
    judgement.verdict = "neutral"
    return judgement


def _build_analysis(context: CoalitionContext) -> StrategicAnalysisResult:
    coalition = CoalitionStabilityAnalysis(
        context=context,
        is_stable=True,
        blocking_coalitions=tuple(),
        shapley_values={"agent_0": 0.7, "agent_1": 0.3},
        core_non_empty=True,
        stability_score=0.5,
        recommendations=tuple(),
    )
    return StrategicAnalysisResult(
        nash_analysis=None,
        coalition_analysis=coalition,
        recommendations=tuple(),
        welfare_metrics={},
        timestamp="now",
        analysis_duration_ms=0.0,
    )


def test_v3_reset_performs_assessment(monkeypatch):
    model = DummyModel()
    context = CoalitionContext(agent_ids=("agent_0", "agent_1"))
    ethics_module = MagicMock()
    strategic_layer = MagicMock()

    ethics_module.judge_distributed.return_value = _build_judgement()
    strategic_layer.analyze.return_value = _build_analysis(context)

    monkeypatch.setattr(
        "erisml.interop.pettingzoo_adapter.ErisEngine", lambda m: MagicMock(model=m)
    )

    env = ErisPettingZooEnv(
        model=model,
        ethics_module=ethics_module,
        strategic_layer=strategic_layer,
        coalition_context=context,
        state_to_facts_fn=_dummy_facts,
    )

    env.reset()

    ethics_module.judge_distributed.assert_called_once()
    strategic_layer.analyze.assert_called_once()


def test_v3_step_calculates_weighted_reward(monkeypatch):
    model = DummyModel()
    context = CoalitionContext(agent_ids=("agent_0", "agent_1"))
    ethics_module = MagicMock()
    strategic_layer = MagicMock()
    engine = MagicMock()
    engine.step.return_value = {"tick": 1}

    judgement = _build_judgement()
    ethics_module.judge_distributed.return_value = judgement
    strategic_layer.analyze.return_value = _build_analysis(context)

    monkeypatch.setattr(
        "erisml.interop.pettingzoo_adapter.ErisEngine", lambda _: engine
    )

    env = ErisPettingZooEnv(
        model=model,
        ethics_module=ethics_module,
        strategic_layer=strategic_layer,
        coalition_context=context,
        state_to_facts_fn=_dummy_facts,
        welfare_weight=1.0,
        stability_weight=1.0,
    )

    env.reset()
    env.step(0)

    expected = judgement.get_party_vector("agent_0").to_scalar() + 0.5
    assert env._cumulative_rewards["agent_0"] == expected

    obs = env.observe("agent_0")
    assert obs["ethical_welfare"] == judgement.get_party_vector("agent_0").to_scalar()
    assert obs["stability_score"] == 0.5
    assert obs["shapley_value"] == 0.7
