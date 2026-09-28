# TraceElephant (урезанная копия)

Источник: https://huggingface.co/datasets/TraceElephant/TraceElephant, файл `data.zip`,
ревизия в `SOURCE_COMMIT`. Лицензия CC-BY-4.0. Статья: Chen et al., «Seeing the
Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent
Systems», arXiv:2604.22708, ACL 2026.

Что изменено. Оригинал весит 830 МБ, из них 250 МБ — `step_records.json`, где на
каждом шаге целиком повторяется весь входной контекст агента. Здесь:

- `trace_metadata.json` скопирован без изменений: там разметка `mistake_agent`,
  `mistake_step`, `mistake_reason`;
- в `step_records.json` у каждого шага удалено поле `input`; `step_id`,
  `agent_id`, `agent_name`, `output`, `tool_logs` сохранены;
- папки `images/` (скриншоты Magentic-One) не скопированы.

Как получено:

```python
import json
from pathlib import Path
src, dst = Path("data"), Path("data/trajectories/traceelephant")  # src — распакованный data.zip
for meta in sorted(src.glob("*/*/trace_metadata.json")):
    run = meta.parent
    out = dst / run.parent.name / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_metadata.json").write_bytes(meta.read_bytes())
    steps = json.loads((run / "step_records.json").read_text())
    (out / "step_records.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "input"} for s in steps], ensure_ascii=False))
```
