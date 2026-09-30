# Data

## Runs of our agent

**`full_cf_stopwords_results.csv`** — 468 counterfactual runs of the agent
with four tools (`agent/tools.py`). In each row, words (`removed_words`) are removed
from the prompt, and the file records whether the agent called the tool `tool` before
(`called_before`) and after (`called_after`) the intervention. `flipped = called_before
and not called_after`. In all rows `called_before = True`.

This is a frozen artifact: the runs were made once, while preparing the
Stepik material on AgentSHAP, and the generation code is not in the repository. Notebook 02
only analyses it. The model and temperature are not recorded in the file.

**`agentshap/`** — results of notebook `03_produce_agentshap_live.ipynb`
(qwen2.5:3b on local Ollama, TokenSHAP `28b2f2c`): coalitions, answers, tool
calls and Shapley values from `analyze()`, noise floor. The saved runs were
produced with the original Russian-language tool outputs (see the `main` branch
history). Rerunning notebook 03 gives different texts and numbers, as any rerun
does: generation is stochastic.

**`counterfactual/replay.csv`** — results of notebook `05_produce_counterfactual_replay.ipynb`:
replay of an AgentDojo trajectory step on qwen2.5:3b with interventions and the full
text of each generation. Generations use a fixed `seed`.

## Trajectories from other benchmarks — `trajectories/`

The source revision for each folder is in `SOURCE_COMMIT`.

| folder | source | license | what was changed |
|---|---|---|---|
| `ata/` | [chenjing-2024/agent-trajectory-attribution](https://github.com/chenjing-2024/agent-trajectory-attribution), `data/cases/`; arXiv:2608.06909 | labels CC-BY-4.0 (`LICENSE-DATA`), trajectories MIT/Apache-2.0 of the source benchmarks (`NOTICE`, `third_party/`) | nothing |
| `hintbench/` | [Jiacheng-31/HINTBench](https://github.com/Jiacheng-31/HINTBench), `data/` | Apache-2.0 (`LICENSE`) | nothing |
| `whowhen/` | [ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution), `Who&When/`; arXiv:2505.00212 | MIT (`LICENSE`) | nothing |
| `traceelephant/` | [TraceElephant on Hugging Face](https://huggingface.co/datasets/TraceElephant/TraceElephant), `data.zip`; arXiv:2604.22708 | CC-BY-4.0 | trimmed from 830 MB: repeated input context removed from steps, screenshots not copied; see `traceelephant/README.md` |

Discrepancies with the source READMEs found while loading:

- HINTBench: `hintbench.json` has 536 trajectories (400 risky, 136 safe),
  not 629 as the README says; labels are in the `risk_labels` field, not `injected_risks`;
  `step_id` has gaps; 154 records have no `env` field; three labels point
  to steps that do not exist. `hintbench_val.json` has no step labels.
- TraceElephant: one trace has `mistake_step = "none"`, another `"Step 36"`.
- agent-trajectory-attribution: component numbering differs between sources;
  Stage has an extra empty system component in `trajectory`.

All reconciliation is done in `bench/load.py`.
