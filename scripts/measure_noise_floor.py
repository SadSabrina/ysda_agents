#!/usr/bin/env python3
"""
Померить noise floor агента и сохранить результат.

Запускать НА ТОЙ ЖЕ МАШИНЕ, где работает Ollama — она слушает localhost,
и из контейнера или чужой VM её обычно не видно.

    ollama serve
    ollama pull qwen2.5:3b
    python scripts/measure_noise_floor.py

Результат пишется в data/noise_floor.json, ноутбук 03 его оттуда подхватывает,
чтобы не гонять модель заново.

Зачем это нужно: baseline в AgentSHAP зашит как 1.0, но temperature=0.5
захардкожена в библиотеке, поэтому два прогона с полным тулсетом не совпадают.
Настоящий потолок близости nu < 1, а разница (1 - nu) — это постоянное
смещение, которое и ломает аксиому dummy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import build_agent_shap, measure_noise_floor_multi  # noqa: E402

DEFAULT_PROMPTS = [
    "What's the weather in New York and how is AAPL stock doing today?",
    "What's the latest AI news and what is 17 plus 25?",
    "How is TSLA doing, and what's the weather in Tokyo?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10,
                    help="прогонов на промпт (по умолчанию 10)")
    ap.add_argument("--model", default=None, help="имя модели в Ollama")
    ap.add_argument("--url", default=None,
                    help="адрес Ollama; по умолчанию ищется автоматически")
    ap.add_argument("--out", default=str(ROOT / "data" / "noise_floor.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    kwargs = {}
    if args.model:
        kwargs["model_name"] = args.model
    if args.url:
        kwargs["url"] = args.url

    try:
        bundle = build_agent_shap(**kwargs)
    except RuntimeError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("Если Ollama работает, но скрипт её не видит — запустите его "
              "на той же машине, где она слушает.", file=sys.stderr)
        return 1

    n_prompts = len(DEFAULT_PROMPTS)
    print(f"\n{n_prompts} промпта × {args.runs} прогонов = "
          f"{n_prompts * args.runs} вызовов модели\n")

    nf = measure_noise_floor_multi(bundle.agent_shap, DEFAULT_PROMPTS,
                                   n_runs=args.runs, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print(nf)
    print("=" * 60)
    print(f"\nожидаемый паразитный вклад на инструмент ~ (1-nu)/n = "
          f"{nf.expected_dummy_share(len(bundle.tools)):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "nu": nf.nu,
        "sd": nf.sd,
        "bias": nf.bias,
        "n_runs": nf.n_runs,
        "behavioural_noise": nf.behavioural_noise,
        "n_tools": len(bundle.tools),
        "model": bundle.model.model_name,
        "prompts": DEFAULT_PROMPTS,
        "samples": nf.samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
