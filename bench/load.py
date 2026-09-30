"""
Four trajectory datasets in one schema.

Case.steps     — trajectory steps in order;
Case.target    — index of the step whose behavior we explain, or None if
                 we explain the outcome of the whole trajectory (failure, risk);
Case.causes    — gold indices of the cause steps;
Case.chain     — gold chain (agent-trajectory-attribution only).

All indices are positions in Case.steps, zero-based. The sources number steps
differently; they are unified here and only here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "trajectories"


@dataclass
class Step:
    role: str
    text: str
    flags: dict = field(default_factory=dict)


@dataclass
class Case:
    source: str
    subset: str
    case_id: str
    kind: str                       # behavior | failure | risk
    steps: list[Step]
    target: int | None
    causes: list[int]
    chain: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def candidates(self) -> list[int]:
        """Steps among which the cause is searched: everything before the target, or the whole trajectory."""
        return list(range(self.target if self.target is not None else len(self.steps)))


def load_ata() -> list[Case]:
    """agent-trajectory-attribution: AgentDojo, Agent3Sigma Stage and Canary.

    Component numbering differs across sources: integers from 1 in AgentDojo,
    'C0'.. in Canary, 'C1'.. in Stage, where the trajectory also has an extra empty
    system component at the start. So ids are matched through
    components_metadata, not by position in the list.
    """
    cases = []
    for meta_path in sorted(DATA.glob("ata/*/*/*/metadata.json")):
        d = meta_path.parent
        t = json.loads((d / "trajectory.json").read_text())
        a = json.loads((d / "annotation.json").read_text())
        cm = t["components_metadata"]
        raw = t["trajectory"][len(t["trajectory"]) - len(cm):]
        pos = {str(c["component_id"]).removeprefix("C"): i for i, c in enumerate(cm)}
        idx = lambda cid: pos[str(cid).removeprefix("C")]
        primary = a["primary_attribution"]["component_id"]
        cases.append(Case(
            source="ata", subset=f"{t['source_benchmark']}/{t['target_type']}",
            case_id=t["case_id"], kind="behavior",
            steps=[Step(s["role"], s["content"],
                        {"injection_marker": c.get("contains_injection_marker", False),
                         "function": c.get("function")})
                   for s, c in zip(raw, cm)],
            target=idx(a["target"]["component_id"]),
            causes=[idx(primary)],
            chain=sorted(idx(c) for c in a["chains"]["attack"] + a["chains"]["execution"]),
            meta={"attributes": a["attributes"], "source_metadata": t["source_metadata"],
                  "confidence": a["quality"]["confidence"]["overall"]},
        ))
    return cases


def load_hintbench() -> list[Case]:
    """HINTBench, test split: internal risks without attacks.

    step_id has gaps (the authors removed some steps and kept the ids),
    so labels are matched by step_id. Labels on steps that do not exist
    are dropped. Not every record has the env field; the domain is taken from the
    task_id prefix. The validation split has no step labels and is not loaded.
    """
    cases = []
    for x in json.loads((DATA / "hintbench" / "hintbench.json").read_text()):
        pos = {s["step_id"]: i for i, s in enumerate(x["trajectory"])}
        labelled = {l.get("risk_origin_step", l.get("step_id")) for l in x["risk_labels"]}
        cases.append(Case(
            source="hintbench", subset=x["task_id"].split("_")[0], case_id=x["task_id"], kind="risk",
            steps=[Step(s["role"], s.get("content") or f"{s.get('thought', '')}\n{s.get('action', '')}")
                   for s in x["trajectory"]],
            target=None,
            causes=sorted(pos[s] for s in labelled if s in pos),
            meta={"is_risky": x["is_risky"], "risk_labels": x["risk_labels"],
                  "dropped_labels": len(labelled - pos.keys())},
        ))
    return cases


def load_whowhen() -> list[Case]:
    """Who&When: failures of multi-agent systems. mistake_step is a zero-based index into history."""
    cases = []
    for subset in ("Hand-Crafted", "Algorithm-Generated"):
        for p in sorted((DATA / "whowhen" / subset).glob("*.json"), key=lambda p: int(p.stem)):
            w = json.loads(p.read_text())
            cases.append(Case(
                source="whowhen", subset=subset, case_id=f"{subset}/{p.stem}", kind="failure",
                steps=[Step(h.get("name", h["role"]), h["content"]) for h in w["history"]],
                target=None, causes=[int(w["mistake_step"])],
                meta={"mistake_agent": w["mistake_agent"], "mistake_reason": w["mistake_reason"]},
            ))
    return cases


def load_traceelephant() -> list[Case]:
    """TraceElephant: fully observed MAS traces. step_id starts at one.

    Two traces have a non-numeric mistake_step: 'none' is skipped,
    'Step 36' is read as 36.
    """
    cases = []
    for meta_path in sorted(DATA.glob("traceelephant/*/*/trace_metadata.json")):
        m = json.loads(meta_path.read_text())
        mistake = m["mistake_step"].removeprefix("Step ").strip()
        if not mistake.isdigit():
            continue
        records = json.loads((meta_path.parent / "step_records.json").read_text())
        steps = [Step(r["agent_name"], (r["output"]["choices"][0]["message"]["content"] or "")
                      if isinstance(r["output"], dict) and "choices" in r["output"] else str(r["output"]))
                 for r in records]
        pos = {r["step_id"]: i for i, r in enumerate(records)}
        cases.append(Case(
            source="traceelephant", subset=meta_path.parent.parent.name, case_id=m["run_id"],
            kind="failure", steps=steps, target=None, causes=[pos[int(mistake)]],
            meta={"mistake_agent": m["mistake_agent"], "system": m["system_name"]},
        ))
    return cases


def load_all() -> list[Case]:
    return load_ata() + load_hintbench() + load_whowhen() + load_traceelephant()
