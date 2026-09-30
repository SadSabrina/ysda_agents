# TraceElephant (trimmed copy)

Source: https://huggingface.co/datasets/TraceElephant/TraceElephant, file `data.zip`,
revision in `SOURCE_COMMIT`. License CC-BY-4.0. Paper: Chen et al., "Seeing the
Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent
Systems", arXiv:2604.22708, ACL 2026.

What was changed. The original is 830 MB, of which 250 MB is `step_records.json`, where
every step repeats the agent's entire input context. Here:

- `trace_metadata.json` is copied unchanged: it holds the labels `mistake_agent`,
  `mistake_step`, `mistake_reason`;
- in `step_records.json` the `input` field is removed from every step; `step_id`,
  `agent_id`, `agent_name`, `output`, `tool_logs` are kept;
- the `images/` folders (Magentic-One screenshots) are not copied.

How it was produced:

```python
import json
from pathlib import Path
src, dst = Path("data"), Path("data/trajectories/traceelephant")  # src is the unpacked data.zip
for meta in sorted(src.glob("*/*/trace_metadata.json")):
    run = meta.parent
    out = dst / run.parent.name / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_metadata.json").write_bytes(meta.read_bytes())
    steps = json.loads((run / "step_records.json").read_text())
    (out / "step_records.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "input"} for s in steps], ensure_ascii=False))
```
