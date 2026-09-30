"""
Benchmark for attributing steps of an agent trajectory.

    from bench import load_all, evaluate, baselines
    cases = load_all()
    evaluate(cases, {"my method": my_method, **baselines.ALL})
"""
from . import baselines
from .load import Case, Step, load_all, load_ata, load_hintbench, load_traceelephant, load_whowhen
from .metrics import evaluate, ranks
from .stats import bootstrap_ci, bootstrap_order_prob, min_repeats, wilson_ci

__all__ = ["Case", "Step", "load_all", "load_ata", "load_hintbench", "load_whowhen",
           "load_traceelephant", "evaluate", "ranks", "baselines",
           "wilson_ci", "bootstrap_ci", "bootstrap_order_prob", "min_repeats"]
