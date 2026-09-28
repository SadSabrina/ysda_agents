# Данные

## Прогоны нашего агента

**`full_cf_stopwords_results.csv`** — 468 контрфактуальных запусков агента
с четырьмя инструментами (`agent/tools.py`). В каждой строке из промпта убраны
слова (`removed_words`) и записано, вызывал ли агент инструмент `tool` до
(`called_before`) и после (`called_after`) вмешательства. `flipped = called_before
and not called_after`. Во всех строках `called_before = True`.

Это замороженный артефакт: прогоны сделаны один раз, в ходе подготовки
Stepik-материала по AgentSHAP, и кода генерации в репозитории нет. Ноутбук 02
только анализирует его. Модель и температура в файле не записаны.

**`agentshap/`** — результаты ноутбука `03_produce_agentshap_live.ipynb`
(qwen2.5:3b на локальной Ollama, TokenSHAP `28b2f2c`): коалиции, ответы, вызовы
инструментов и значения Шепли `analyze()`, noise floor. Перезапуск 03 даёт
другие числа: генерация стохастична.

**`counterfactual/replay.csv`** — результаты ноутбука `05_produce_counterfactual_replay.ipynb`:
повтор шага траекторий AgentDojo на qwen2.5:3b с вмешательствами и полным
текстом каждой генерации. Генерации с фиксированным `seed`.

## Траектории чужих бенчмарков — `trajectories/`

Ревизия исходника в каждой папке в `SOURCE_COMMIT`.

| папка | источник | лицензия | что изменено |
|---|---|---|---|
| `ata/` | [chenjing-2024/agent-trajectory-attribution](https://github.com/chenjing-2024/agent-trajectory-attribution), `data/cases/`; arXiv:2608.06909 | разметка CC-BY-4.0 (`LICENSE-DATA`), траектории — MIT/Apache-2.0 исходных бенчмарков (`NOTICE`, `third_party/`) | ничего |
| `hintbench/` | [Jiacheng-31/HINTBench](https://github.com/Jiacheng-31/HINTBench), `data/` | Apache-2.0 (`LICENSE`) | ничего |
| `whowhen/` | [ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution), `Who&When/`; arXiv:2505.00212 | MIT (`LICENSE`) | ничего |
| `traceelephant/` | [TraceElephant на Hugging Face](https://huggingface.co/datasets/TraceElephant/TraceElephant), `data.zip`; arXiv:2604.22708 | CC-BY-4.0 | урезано с 830 МБ: из шагов удалён повторяющийся входной контекст, скриншоты не скопированы; см. `traceelephant/README.md` |

Расхождения с README исходников, найденные при загрузке:

- HINTBench: в `hintbench.json` 536 траекторий (400 рискованных, 136 безопасных),
  а не 629, как в README; разметка в поле `risk_labels`, а не `injected_risks`;
  `step_id` идёт с пропусками; у 154 записей нет поля `env`; три метки ссылаются
  на несуществующие шаги. В `hintbench_val.json` разметки шагов нет.
- TraceElephant: у одной трассы `mistake_step = "none"`, у другой `"Step 36"`.
- agent-trajectory-attribution: нумерация компонентов различается по источникам,
  у Stage в `trajectory` лишний пустой system-компонент.

Все сведения сделаны в `bench/load.py`.
