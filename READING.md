# What to read

For the lecture "Sufficiency vs necessity", YSDA, 30.09.2026.

Sections follow the order of the lecture. Each work has a short note on why
to read it. All links, numbers and quotes were checked against the primary sources on 28.09.2026.
Dates of arXiv preprints are of the first version; "preprint" means that publication
at a conference or in a journal is not confirmed.

**If you read only three:**
1. Watson et al., 2021 — why SHAP measures sufficiency.
2. Pearl, 1999 — what necessity and sufficiency are and when they can be computed.
3. Nasr et al., 2025 — why no reported defense number can be trusted without an adaptive attack.

---

## 1. Necessity, sufficiency, Shapley value

**J. Pearl. "Probabilities of causation: three counterfactual interpretations
and their identification".** Synthese 121, 93–149, 1999.
[doi:10.1023/A:1005233831499](https://doi.org/10.1023/A:1005233831499),
[PDF](https://ftp.cs.ucla.edu/pub/stat_ser/r260-reprint.pdf).
Definitions of PN, PS and PNS (defs. 7–9), Lemma 1
$\text{PNS} = P(x,y)\,\text{PN} + P(x',y')\,\text{PS}$, bounds on PN and PS
from experimental data, and the conditions under which they collapse to a point
(exogeneity and monotonicity). PN and PS are different quantities, one cannot be derived
from the other, but they are linked by Lemma 1. Notebook 01 is built on this paper.

**J. Pearl. "Causality: Models, Reasoning, and Inference",** 2nd ed.,
Cambridge University Press, 2009, ch. 9 "Probability of causation:
interpretation and identification", pp. 283–308.
[doi:10.1017/CBO9780511803161](https://doi.org/10.1017/CBO9780511803161).
The same material as a book chapter, with the general machinery of structural models.
Journal version with the bounds: J. Tian, J. Pearl, "Probabilities of causation:
bounds and identification", Annals of Mathematics and AI 28, 287–313, 2000,
[PDF](https://ftp.cs.ucla.edu/pub/stat_ser/r271-A.pdf).

**D. S. Watson, L. Gultchin, A. Taly, L. Floridi. "Local explanations via
necessity and sufficiency: unifying theory and practice".** UAI 2021,
PMLR 161:1382–1392. [arXiv:2103.14651](https://arxiv.org/abs/2103.14651).
Proposition 1: the SHAP value function $v(S) = \mathbb{E}[f(x_S, X_R)]$ equals
the probability of sufficiency of a feature subset. Hence the main idea of the
lecture: the Shapley value is the average gain in sufficiency, and by construction
it says nothing about necessity. Read before analysing any SHAP-like method.

**L. S. Shapley. "A value for n-person games".** In: Contributions to the
Theory of Games, vol. II, Princeton University Press, 1953, pp. 307–317.
[doi:10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018).
Four axioms and uniqueness of the solution. Short, worth reading
in the original: the axioms there are a contract. If an implementation violates an axiom,
what it computes is no longer the Shapley value.

**B. Bilodeau, N. Jaques, P. W. Koh, B. Kim. "Impossibility theorems for
feature attribution".** PNAS 121(2), 2024.
[doi:10.1073/pnas.2304406120](https://doi.org/10.1073/pnas.2304406120),
[arXiv:2212.11870](https://arxiv.org/abs/2212.11870).
Theorem 3.3: for a complete and linear attribution method (SHAP, Integrated
Gradients), specificity ≤ 1 − sensitivity, i.e. for inferring
local model behaviour it is no better than random guessing. A limit of a class of methods,
not a complaint about implementations.

**S. Wachter, B. Mittelstadt, C. Russell. "Counterfactual explanations without
opening the black box: automated decisions and the GDPR".** 2017.
[arXiv:1711.00399](https://arxiv.org/abs/1711.00399).
A counterfactual explanation in the XAI sense: the nearest input with a different answer.
Useful so as not to confuse it with Pearl's counterfactual (the same case,
a different cause).

## 2. Attribution for agents

**M. Horovicz. "AgentSHAP: Interpreting LLM agent tool importance with Monte
Carlo Shapley value estimation".** Preprint, 14.12.2025.
[arXiv:2512.12597](https://arxiv.org/abs/2512.12597).
The method analysed in notebook 02: Shapley values over the agent's tools,
$v(S)$ is the similarity of the answer with tool set $S$ to the answer with the full set.
Experiments on 8 API-Bank tools, GPT-4o-mini. The code is inside the
[GenAISHAP/TokenSHAP](https://github.com/GenAISHAP/TokenSHAP) repository.
The paper calls itself the "first framework for explaining tool importance in LLM
agents", but ShapleyFlow (below) came out ten months earlier.

**M. Horovicz, R. Goldshmidt. "TokenSHAP: Interpreting large language models
with Monte Carlo Shapley value estimation".** NLP4Science Workshop, 2024,
pp. 1–8. [doi:10.18653/v1/2024.nlp4science-1.1](https://doi.org/10.18653/v1/2024.nlp4science-1.1).
The predecessor of AgentSHAP: the same thing over prompt tokens.

**Y. Yang, B. Huang, S. Qi et al. "Understanding and optimizing agentic
workflows via Shapley value"** (ShapleyFlow). Preprint, 01.02.2025.
[arXiv:2502.00510](https://arxiv.org/abs/2502.00510).
Shapley values over the components of an agentic workflow.

**C. Liu, Y. Zhang, Y. Zhong et al. "SkillShapley: Boundary-adaptive Shapley
valuation for skill step attribution in LLM agents".** Preprint, 13.08.2026.
[arXiv:2608.13173](https://arxiv.org/abs/2608.13173).
Attribution over skill steps instead of tools, two-phase adaptive
sampling of coalitions. Experiments on three SkillsBench tasks (a benchmark
with deterministic verifiers). Two kinds of baselines: attribution methods
(Individual, Leave-One-Out, Random Removal, LeastCore, exact Shapley)
and Shapley approximations (Monte Carlo, Quasi-MC, paired MC, truncated by
coalition size). The paper has no link to code. It does not cite AgentSHAP;
it cites TokenSHAP.

**B. Cohen-Wang, H. Shah, K. Georgiev, A. Mądry. "ContextCite: attributing
model generation to context".** NeurIPS 2024.
[arXiv:2409.00729](https://arxiv.org/abs/2409.00729),
code [MadryLab/context-cite](https://github.com/MadryLab/context-cite).
Attributes the answer to context sentences via random ablations
and a linear surrogate. Needs logprobs.

**F. Liu, N. Kandpal, C. Raffel. "AttriBoT: a bag of tricks for efficiently
approximating leave-one-out context attribution".**
[arXiv:2411.15102](https://arxiv.org/abs/2411.15102),
code [r-three/AttriBoT](https://github.com/r-three/AttriBoT).
A fast approximation of leave-one-out, i.e. closer to necessity. Needs logprobs.

**P. R. Aryan, F. J. Ekaputra. "Semantic provenance and policy-aware
explanations for agentic coding systems".** TAAPAAI workshop at ESWC 2026,
CEUR-WS vol. 4210, paper 6. [PDF](https://ceur-ws.org/Vol-4210/paper6.pdf).
So far the only work citing AgentSHAP: it mentions it among
attribution methods and proposes provenance over execution traces
instead. Connects to provenance in Act 2 of the lecture.

*Background: where prompt compression and sensitivity to wording sit next to attribution.*

- H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, L. Qiu. "LLMLingua: compressing prompts
  for accelerated inference of large language models". EMNLP 2023, pp. 13358–13376.
  [doi:10.18653/v1/2023.emnlp-main.825](https://doi.org/10.18653/v1/2023.emnlp-main.825).
  And H. Jiang et al. "LongLLMLingua". ACL 2024, pp. 1658–1677.
  [doi:10.18653/v1/2024.acl-long.91](https://doi.org/10.18653/v1/2024.acl-long.91).
  Not attribution: token importance by the perplexity of a small model, for compression.
  Code and package [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua).
- E. Mosca, F. Szigeti, S. Tragianni, D. Gallagher, G. Groh. "SHAP-based
  explanation methods: a review for NLP interpretability". COLING 2022,
  pp. 4593–4603. [ACL Anthology 2022.coling-1.406](https://aclanthology.org/2022.coling-1.406/).
  A review of SHAP variants for text.
- Z. Jiang, F. F. Xu, J. Araki, G. Neubig. "How can we know what language
  models know?". TACL 8, 423–438, 2020.
  [doi:10.1162/tacl_a_00324](https://doi.org/10.1162/tacl_a_00324).
  The model's answer depends strongly on how the prompt is worded.

## 3. Why ablation lies

**T. McGrath, M. Rahtz, J. Kramár, V. Mikulik, S. Legg. "The Hydra effect:
emergent self-repair in language model computations".** Preprint, 2023.
[arXiv:2307.15771](https://arxiv.org/abs/2307.15771).
If you remove an attention layer, other layers partly compensate for its contribution.
The measured ablation effect underestimates the component's importance.

**C. Rushing, N. Nanda. "Explorations of self-repair in language models".**
ICML 2024. [arXiv:2402.15390](https://arxiv.org/abs/2402.15390).
Self-repair occurs across model families and sizes; it is partly
explained by the final LayerNorm and by sparse sets of neurons.

**D. Janzing, L. Minorics, P. Blöbaum. "Feature relevance quantification in
explainable AI: a causal problem".** AISTATS 2020, PMLR 108:2907–2916.
[arXiv:1910.13413](https://arxiv.org/abs/1910.13413).
What to replace a "removed" feature with: the interventional formulation vs
the observational one.

**S. Hooker, D. Erhan, P.-J. Kindermans, B. Kim. "A benchmark for
interpretability methods in deep neural networks"** (ROAR). NeurIPS 2019.
[arXiv:1806.10758](https://arxiv.org/abs/1806.10758).
Attribution quality via feature removal with model retraining.
Does not transfer to an agent: you cannot retrain it.

**Y. Rong, T. Leemann, V. Borisov, G. Kasneci, E. Kasneci. "A consistent and
efficient evaluation strategy for attribution methods"** (ROAD). ICML 2022.
[arXiv:2202.00449](https://arxiv.org/abs/2202.00449).
The same without retraining; removes information leakage through the shape of the mask.

**S. Krishna, T. Han, A. Gu et al. "The disagreement problem in explainable
machine learning: a practitioner's perspective".** TMLR, 2024.
[arXiv:2202.01602](https://arxiv.org/abs/2202.01602).
Different attribution methods give different answers: on COMPAS, 14 of 15 pairs of methods
have negative rank correlation. 86 % of surveyed practitioners
resolve disagreements with heuristics or do not know how.

## 4. Explanations you cannot trust

**M. Turpin, J. Michael, E. Perez, S. R. Bowman. "Language models don't always
say what they think: unfaithful explanations in chain-of-thought prompting".**
NeurIPS 2023. [arXiv:2305.04388](https://arxiv.org/abs/2305.04388).
A hint in the prompt shifts the answer (accuracy drops by up to 36 %), and the explanation
says not a word about it.

**T. Lanham, A. Chen, A. Radhakrishnan et al. "Measuring faithfulness in
chain-of-thought reasoning".** Preprint, 2023.
[arXiv:2307.13702](https://arxiv.org/abs/2307.13702).
In 7 of 8 tasks the 175B model relies on its CoT less than the 13B model;
the most faithful reasoning in 6 of 8 tasks is from the 13B. The dependence on size
is non-monotonic.

**C.-E. Sun, L. Liu, G. Yan, Z. Wang, T.-W. Weng. "LLM agents already know when
to call tools — even without reasoning".** Preprint, 2026.
[arXiv:2605.09252](https://arxiv.org/abs/2605.09252).
Whether a tool is needed can be linearly decoded from the model's representation before
generation starts: AUROC 0.89–0.96 on six models. The decision is made before
the explanation is written.

**A. P. Gema, N. Rajani, R. Saxena, W.-C. Kwan, P. Minervini.
"Chain-of-thought faithfulness of reasoning models varies with where and how
preference cues are delivered".** Preprint, 2026.
[arXiv:2608.29464](https://arxiv.org/abs/2608.29464).
Models verbalize a hint that arrives through a tool response in their CoT
less often than a hint from the user message, in all 15 open models
tested. Directly about agents: the tool channel is the least explained.

**Y. Gur-Arieh, A. Marasović, M. Geva. "Faithfulness metrics don't measure
faithfulness: a meta-evaluation with ground truth".** Preprint, 2026.
[arXiv:2605.25052](https://arxiv.org/abs/2605.25052).
The BonaFide benchmark: most CoT faithfulness metrics distinguish faithful from
unfaithful reasoning at the level of AUROC ≈ 0.5. The best are CC-SHAP (0.70)
at the level of the whole CoT and Filler Tokens (0.59) at the step level. Read before
you adopt any faithfulness metric.

## 5. Trajectory benchmarks and failure attribution

The four datasets that live in `data/trajectories/` and are used
in notebook 04 (what was changed when copying is in `data/README.md`):

**J. Chen, Y. Sun, L. Zhang, L. Xu, J. Shi. "Long-horizon agent trajectory
attribution: a unified benchmark and fine-grained annotation framework".**
Preprint, 07.08.2026. [arXiv:2608.06909](https://arxiv.org/abs/2608.06909),
data [chenjing-2024/agent-trajectory-attribution](https://github.com/chenjing-2024/agent-trajectory-attribution).
1 351 trajectories from AgentDojo and Agent3Sigma: target action, cause step,
attack and execution chains. The labels were made by an LLM and rules, not by humans.

**HINTBench** — internal risks in long trajectories without attacks.
[Jiacheng-31/HINTBench](https://github.com/Jiacheng-31/HINTBench),
Apache-2.0; the repository has no link to a paper.

**S. Zhang, M. Yin, J. Zhang et al. "Which agent causes task failures and when?
On automated failure attribution of LLM multi-agent systems"** (Who&When).
ICML 2025. [arXiv:2505.00212](https://arxiv.org/abs/2505.00212).
184 failures of multi-agent systems, labelled with the responsible agent and the decisive step.
The best agent-level method (All-at-once) gets 53.5 %, the best step-level one
(Step-by-step) 14.2 % (GPT-4o, averaged over settings).

**M. Chen, J. Wang, F. Mu et al. "Seeing the whole elephant: a benchmark for
failure attribution in LLM-based multi-agent systems"** (TraceElephant).
ACL 2026. [arXiv:2604.22708](https://arxiv.org/abs/2604.22708).
220 fully observable traces and runnable system code, so
counterfactual replays are possible. Annotator agreement: Krippendorff's α
0.72 at the agent level, 0.64 at the step level.

More on failure attribution:

- **M. Cemri, M. Z. Pan, S. Yang et al. "Why do multi-agent LLM systems fail?"**
  (MAST). NeurIPS 2025, Datasets & Benchmarks.
  [arXiv:2503.13657](https://arxiv.org/abs/2503.13657).
  A taxonomy of 14 failure modes, 1 642 traces from 7 frameworks;
  human agreement when building the taxonomy κ = 0.88.
- **D. Deshpande, V. Gangal, H. Mehta et al. "TRAIL: trace reasoning and agentic
  issue localization".** Preprint, 2025.
  [arXiv:2505.08638](https://arxiv.org/abs/2505.08638).
  148 traces, 841 labelled errors; the best model (Gemini-2.5-Pro) gets 11 %.
- **G. Zhang, J. Wang, J. Chen et al. "AgenTracer: who is inducing failure in
  the LLM agentic systems?".** Preprint, 2025.
  [arXiv:2509.03312](https://arxiv.org/abs/2509.03312).
- **F. Kong, R. Zhang, H. Yin et al. "Aegis: automated error generation and
  attribution for multi-agent systems".** ICLR 2026.
  [arXiv:2509.14295](https://arxiv.org/abs/2509.14295). 9 533 trajectories.
- **J. Liu, H. Xi, S. Zhang et al. "Who&When Pro: can LLMs really attribute
  failures in AI agents?".** Preprint, 10.07.2026.
  [arXiv:2607.09996](https://arxiv.org/abs/2607.09996).
  12 326 failed traces with gold labels from 26 benchmarks.
- **Y. Li, H. Luo, Y. Xie et al. "ATBench: a diverse and realistic agent
  trajectory benchmark for safety evaluation and diagnosis".** COLM 2026.
  [arXiv:2604.02022](https://arxiv.org/abs/2604.02022).
  Safety of the trajectory as a whole, without step labels.
- **J. Kim, S. Mun, S. Lee et al. "Towards faithful agentic XAI: a verification
  method and an open-world benchmark for better model faithfulness"**
  (CRAFTER-XAI-Bench). Preprint, 2026.
  [arXiv:2605.27879](https://arxiv.org/abs/2605.27879).
  Tests explainer agents for faithfulness; there SHAP is a tool of the agent,
  not a method being compared.

## 6. Agent security: what is measured and how

- **E. Debenedetti, J. Zhang, M. Balunović et al. "AgentDojo: a dynamic
  environment to evaluate prompt injection attacks and defenses for LLM
  agents".** NeurIPS 2024, Datasets & Benchmarks.
  [arXiv:2406.13352](https://arxiv.org/abs/2406.13352).
- **Q. Zhan, Z. Liang, Z. Ying, D. Kang. "InjecAgent: benchmarking indirect
  prompt injections in tool-integrated LLM agents".** Findings of ACL 2024.
  [arXiv:2403.02691](https://arxiv.org/abs/2403.02691).
- **H. Zhang, J. Huang, K. Mei et al. "Agent Security Bench (ASB)".** ICLR 2025.
  [arXiv:2410.02644](https://arxiv.org/abs/2410.02644).
- **Agent3Sigma** — a three-level evaluation of agent security (static,
  simulation, real tools). [antgroup/Agent3Sigma](https://github.com/antgroup/Agent3Sigma).

Numbers across these benchmarks are not directly comparable: each has its own
setup.

**M. Dziemian, M. Lin, X. Fu et al. "How vulnerable are AI agents to indirect
prompt injections? Insights from a large-scale public competition".**
Preprint, 2026. [arXiv:2603.15714](https://arxiv.org/abs/2603.15714).
41 scenarios, 13 models; attack success rate from 0.5 % to 8.5 % depending
on the model, a spread of more than an order of magnitude. The main argument for measurement
over intuition.

**A. Zou, M. Lin, E. Jones et al. "Security challenges in AI agent deployment:
insights from a large scale public competition".** Preprint, 2025.
[arXiv:2507.20526](https://arxiv.org/abs/2507.20526).
1.8 M attacks on 22 agents in 44 scenarios, more than 60 000 successful.

**M. Nasr, N. Carlini, C. Sitawarin et al. "The attacker moves second:
stronger adaptive attacks bypass defenses against LLM jailbreaks and prompt
injections".** Preprint, 2025. [arXiv:2510.09023](https://arxiv.org/abs/2510.09023).
Adaptive attacks bypassed 12 recent defenses, most with success above 90 %,
although most of these defenses had reported near-zero attack success. Read
before you believe any defense number.

## 7. Provenance

**D. E. Denning. "A lattice model of secure information flow".**
Communications of the ACM 19(5), 236–243, 1976.
[doi:10.1145/360051.360056](https://doi.org/10.1145/360051.360056).
Where labels and their propagation come from: a lattice of security levels.

**Taint mode in Perl** — [perlsec](https://perldoc.perl.org/perlsec).
The clearest example: data from outside is labelled, and if labelled data reaches
`exec` without a check, the program dies.

## 8. How interpretability is validated in general

A map of the field, if you want to dig beyond attribution.

| Family | Canonical works | Known criticism |
|---|---|---|
| Feature attribution | ERASER (DeYoung et al., ACL 2020, [arXiv:1911.03429](https://arxiv.org/abs/1911.03429)); ROAR; ROAD | removal pushes the input out of distribution |
| Sanity checks | Adebayo et al., "Sanity checks for saliency maps", NeurIPS 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292) | some methods cannot tell a trained network from a random one |
| Attention as explanation | Jain & Wallace, NAACL 2019, [arXiv:1902.10186](https://arxiv.org/abs/1902.10186); Wiegreffe & Pinter, EMNLP 2019, [arXiv:1908.04626](https://arxiv.org/abs/1908.04626) | Bastings & Filippova, 2020, [arXiv:2010.05607](https://arxiv.org/abs/2010.05607): the dispute cannot be settled without a definition of explanation |
| Probing | Hewitt & Liang, control tasks, EMNLP 2019, [arXiv:1909.03368](https://arxiv.org/abs/1909.03368); Elazar et al., amnesic probing, TACL 2021, [arXiv:2006.00995](https://arxiv.org/abs/2006.00995) | probe accuracy is not the same as the feature being used |
| Activation patching | Zhang & Nanda, ICLR 2024, [arXiv:2309.16042](https://arxiv.org/abs/2309.16042) | sensitivity to the choice of corrupted distribution |
| Circuits | IOI (Wang et al., [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)); ACDC (Conmy et al., NeurIPS 2023, [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)); MIB (Mueller et al., ICML 2025, [arXiv:2504.13151](https://arxiv.org/abs/2504.13151)) | a lot of manual work, scales poorly |
| SAE | SAEBench (Karvonen et al., ICML 2025, [arXiv:2503.09532](https://arxiv.org/abs/2503.09532)); SynthSAEBench ([arXiv:2602.14687](https://arxiv.org/abs/2602.14687)) | feature absorption; proxy metrics do not transfer |
| Concept-based | TCAV (Kim et al., ICML 2018, [arXiv:1711.11279](https://arxiv.org/abs/1711.11279)) | concept leakage between layers |
| Counterfactuals | CARLA (Pawelczyk et al., NeurIPS 2021, [arXiv:2108.00783](https://arxiv.org/abs/2108.00783)) | the explanation can be gamed |
| CoT faithfulness | BonaFide (section 4) | most metrics at the level of AUROC ≈ 0.5 |
| Agent level | CRAFTER-XAI-Bench, TraceElephant (section 5) | the family is still taking shape |

Five failure modes that recur across all families:

1. The intervention pushes the input out of distribution.
2. Validation is circular: the metric coincides with what the method optimizes.
3. There is no ground truth: no labels saying "this is what actually had an influence".
4. The metric can be fooled: the explanation is tuned without changing the model.
5. The mechanism compensates: a removed component is replaced by another.

Knowing these five is more useful than knowing any single method. You can't learn them all anyway = )
