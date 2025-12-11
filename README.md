# ErisML Library

ErisML is a modeling language for governed, foundation-model-enabled agents
operating in pervasive computing environments (homes, hospitals, campuses,
factories, vehicles, etc.).

ErisML provides a single, machine-interpretable and human-legible representation of

- **(i) environment state and dynamics**
- **(ii) agents and their capabilities and beliefs**
- **(iii) intents and utilities**
- **(iv) norms (permissions, obligations, prohibitions, sanctions)**
- **(v) multi-agent strategic interaction**

We define a concrete syntax, a formal grammar, denotational semantics, and
an execution model that treats norms as first-class constraints on action,
introduces longitudinal safety metrics such as Norm Violation Rate (NVR) and
Alignment Drift Velocity (ADV), and supports compilation to planners, verifiers, and simulators.

On top of this, ErisML now includes an **ethics-only decision layer (DEME)**:

- A structured **EthicalFacts** abstraction used as the *only* input to ethics modules.
- Pluggable **EthicsModule** implementations that perform purely normative reasoning.
- A **democratic governance** layer that aggregates multiple EthicalJudgement outputs.
- A worked example for **clinical triage under resource scarcity**.

![CI](https://github.com/ahb-sjsu/erisml-lib/actions/workflows/ci.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a production-style Python library with:

- Modern `src/` layout and `pyproject.toml`
- GitHub Actions CI using:
  - **Black 24.4.2** for formatting checks
  - **Ruff** for linting
  - **Taplo** for TOML validation
  - **Pytest** for tests
  - A **DEME smoke test** that runs the triage ethics demo
- Core ErisML implementation:
  - Language grammar (Lark)
  - Typed AST (Pydantic)
  - Core IR (environment, agents, norms)
  - Runtime engine with a norm gate and metrics
  - PettingZoo adapter for multi-agent RL
  - PDDL/tarski adapter stub for planning
- **Ethics / DEME subsystem**:
  - Structured **EthicalFacts** and ethical dimensions (consequences, rights/duties, fairness, autonomy, privacy, societal/environmental, etc.)
  - **EthicalJudgement** and **EthicsModule** interface
  - Governance config and aggregation (`GovernanceConfig`, `DecisionOutcome`, `select_option`)
  - A **Case Study 1 triage module** (`CaseStudy1TriageEM`) and a small rights-first EM
- Executable examples:
  - **TinyHome** norm-gated environment
  - **Triage ethics demo** combining multiple EMs under governance
- A basic test suite

## Quickstart (Windows / PowerShell)

```powershell
cd erisml-lib

python -m venv .venv
.\.venv\Scripts\activate

pip install -e ".[dev]"

pytest
