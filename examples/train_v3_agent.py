"""Demo: run a tiny governance-aware loop with the V3 PettingZoo adapter."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from erisml.core.model import AgentModel, EnvironmentModel, ErisModel
from erisml.core.types import EnvironmentRule
from erisml.ethics.coalition import CoalitionContext
from erisml.ethics.facts_v3 import (
    ConsequencesV3,
    EthicalFactsV3,
    JusticeAndFairnessV3,
    PartyConsequences,
    PartyJustice,
    PartyRights,
    RightsAndDutiesV3,
)
from erisml.ethics.layers.strategic import StrategicLayer
from erisml.ethics.modules.tier0.geneva_em_v3 import GenevaEMV3
from erisml.interop.pettingzoo_adapter import ErisPettingZooEnv


def state_to_facts_stub(state: dict) -> EthicalFactsV3:
    del state
    return EthicalFactsV3(
        option_id="demo_step",
        consequences=ConsequencesV3(
            expected_benefit=0.8,
            expected_harm=0.1,
            urgency=0.5,
            affected_count=2,
            per_party=(
                PartyConsequences(
                    party_id="agent_0",
                    expected_benefit=0.8,
                    expected_harm=0.1,
                ),
                PartyConsequences(
                    party_id="agent_1",
                    expected_benefit=0.7,
                    expected_harm=0.2,
                ),
            ),
        ),
        rights_and_duties=RightsAndDutiesV3(
            violates_rights=False,
            has_valid_consent=True,
            violates_explicit_rule=False,
            role_duty_conflict=False,
            per_party=(
                PartyRights(party_id="agent_0", rights_violated=False),
                PartyRights(party_id="agent_1", rights_violated=False),
            ),
        ),
        justice_and_fairness=JusticeAndFairnessV3(
            discriminates_on_protected_attr=False,
            prioritizes_most_disadvantaged=True,
            per_party=(
                PartyJustice(party_id="agent_0", relative_burden=0.3),
                PartyJustice(party_id="agent_1", relative_burden=0.4),
            ),
        ),
    )


def main() -> None:
    def increment_tick(state: dict, params: dict) -> dict:
        del params
        next_tick = int(state.get("tick", 0)) + 1
        return {"tick": next_tick}

    env_model = EnvironmentModel(name="ResourceAllocation")
    env_model.add_rule(
        EnvironmentRule(name="advance", param_names=[], update_fn=increment_tick)
    )

    model = ErisModel(
        env=env_model,
        agents={
            "agent_0": AgentModel(name="agent_0"),
            "agent_1": AgentModel(name="agent_1"),
        },
    )

    env = ErisPettingZooEnv(
        model=model,
        ethics_module=GenevaEMV3(),
        strategic_layer=StrategicLayer(),
        coalition_context=CoalitionContext(agent_ids=("agent_0", "agent_1")),
        state_to_facts_fn=state_to_facts_stub,
        welfare_weight=1.0,
        stability_weight=0.5,
    )

    print("Starting training loop (demo)...")
    env.reset()
    for _ in range(10):
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)
            reward = env._cumulative_rewards[agent]
            print(f"Agent: {agent}, Reward: {reward:.4f}")

    env.close()
    print("Training demo finished.")


if __name__ == "__main__":
    main()
