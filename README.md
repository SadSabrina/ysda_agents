# Sufficiency vs necessity

Materials for a guest lecture at the YSDA (Yandex School of Data Analysis) intensive, 30 September 2026.

A continuation of the talk by Lecturer 1 (day 1 of the intensive, 29.09). It showed what must
hold in an agent whatever the model answers, and how to check it with a stub test
(the model replaced by a stub). Here: how to check the system from the outside, when the check
is not a guarantee but a signal.

## Main idea

When people ask "what influenced the model's answer", they mean
**necessity**: without what would the answer have been different. Attribution methods (SHAP,
LIME, occlusion) by their math measure **sufficiency** and need
many runs before they can claim anything. This is not an approximation of necessity,
it is a different quantity.

## Notebooks

| | act | what's inside | what you need |
|---|---|---|---|
| [01](notebooks/01_necessity_vs_sufficiency.ipynb) | 0. Bridge | PN, PS, PNS; redundancy on the EchoLeak example; PN from an experiment is only an interval; iris: Shapley vs necessity | From logs |
| [02](notebooks/02_agent_attribution.ipynb) | 1. How agents are explained | what exists out of the box; Critique A (AgentSHAP implementation: degeneracy, baseline, dummy); Critique B (Shapley measures sufficiency); how many runs are needed | From logs |
| [03](notebooks/03_produce_agentshap_live.ipynb) | for Act 1 | live AgentSHAP run, writes `data/agentshap/` | Ollama + time for the experiment |
| [04](notebooks/04_trajectory_benchmark.ipynb) | 2. Trajectory benchmarking | four datasets in one schema, trivial baselines and leakage, counterfactual check of the labels, provenance, the ladder, a template for your own method | From logs |
| [05](notebooks/05_produce_counterfactual_replay.ipynb) | for Act 2 | counterfactual replay of AgentDojo trajectories, writes `data/counterfactual/` | Ollama + time for the experiment |

01, 02 and 04 work offline on saved data. 03 and 05 run the model
and were run in advance; their results are in the repository.

## Code

```
agent/        agent on Ollama + AgentSHAP (TokenSHAP is cloned on first run),
              per-tool attribution, noise floor
bench/        loading of four trajectory datasets, metrics, trivial baselines,
              statistics (intervals, number of runs), counterfactual replay
data/         see data/README.md: provenance, licenses, what was changed
READING.md    what to read, with notes on why
```

## Running

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
jupyter lab notebooks/01_necessity_vs_sufficiency.ipynb
```

For 03 and 05 you need a local model, no paid API:

```bash
ollama serve
ollama pull qwen2.5:3b
```

## Main takeaway from the talk

Today (September 2026), explaining an agent is not theoretical XAI but rather a ladder of trade-offs between "reliable" and "hard". Numbering the rungs, it can be described like this:

| | what you do | runs | what you get |
|---|---|---|---|
| 1 | invariant | 1, with a stub | holds whatever the model answers |
| 2 | provenance | 1, deterministic | "could have influenced" by construction, with over-blocking |
| 3 | counterfactual | tens to hundreds | measured necessity with an interval |
| 4 | attribution | 2ⁿ | a number that answers a different question |

Go down a rung only when the one above is impossible. 

Whatever you measure, report:

1. **How many runs** stand behind the claim? 
2. **Interval?** As we saw in the slides, 0 hits out of 3 is not 0 % but [0 %, 56 %].
3. **Did you check the counterfactual?** Without this link, would $y$ have happened?
4. **Is there redundancy?** If two rules always fire together,
   the importance of each is undefined for any amount of data.

```python
from bench import load_all, evaluate, baselines, wilson_ci, min_repeats
```
