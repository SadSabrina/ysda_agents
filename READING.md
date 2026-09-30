# Что почитать

К лекции «Достаточность против необходимости», ШАД, 30.09.2026.

Разделы идут в порядке лекции. У каждой работы коротко сказано, зачем её
читать. Все ссылки, числа и цитаты сверены с первоисточниками 28.09.2026.
Даты у arXiv-препринтов — первая версия; «препринт» значит, что публикация
на конференции или в журнале не подтверждена.

**Если читать только три:**
1. Watson et al., 2021 — почему SHAP меряет достаточность.
2. Pearl, 1999 — что такое необходимость и достаточность и когда их можно посчитать.
3. Nasr et al., 2025 — почему любой заявленной цифре защиты нельзя верить без адаптивной атаки.

---

## 1. Необходимость, достаточность, значение Шепли

**J. Pearl. «Probabilities of causation: three counterfactual interpretations
and their identification».** Synthese 121, 93–149, 1999.
[doi:10.1023/A:1005233831499](https://doi.org/10.1023/A:1005233831499),
[PDF](https://ftp.cs.ucla.edu/pub/stat_ser/r260-reprint.pdf).
Определения PN, PS и PNS (опр. 7–9), лемма 1
$\text{PNS} = P(x,y)\,\text{PN} + P(x',y')\,\text{PS}$, границы для PN и PS
по экспериментальным данным и условия, при которых они сжимаются в точку
(экзогенность и монотонность). PN и PS — разные величины, одна не выводится
из другой, но они связаны леммой 1. По этой статье построен ноутбук 01.

**J. Pearl. «Causality: Models, Reasoning, and Inference»,** 2-е изд.,
Cambridge University Press, 2009, гл. 9 «Probability of causation:
interpretation and identification», с. 283–308.
[doi:10.1017/CBO9780511803161](https://doi.org/10.1017/CBO9780511803161).
Тот же материал в виде главы книги, с общим аппаратом структурных моделей.
Журнальная версия с границами: J. Tian, J. Pearl, «Probabilities of causation:
bounds and identification», Annals of Mathematics and AI 28, 287–313, 2000,
[PDF](https://ftp.cs.ucla.edu/pub/stat_ser/r271-A.pdf).

**D. S. Watson, L. Gultchin, A. Taly, L. Floridi. «Local explanations via
necessity and sufficiency: unifying theory and practice».** UAI 2021,
PMLR 161:1382–1392. [arXiv:2103.14651](https://arxiv.org/abs/2103.14651).
Proposition 1: функция ценности SHAP $v(S) = \mathbb{E}[f(x_S, X_R)]$ равна
вероятности достаточности подмножества признаков. Отсюда главная мысль
лекции: значение Шепли — средний прирост достаточности, а про необходимость
оно молчит по построению. Читать до разбора любого SHAP-подобного метода.

**L. S. Shapley. «A value for n-person games».** В кн.: Contributions to the
Theory of Games, vol. II, Princeton University Press, 1953, с. 307–317.
[doi:10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018).
Четыре аксиомы и единственность решения. Короткая, стоит прочитать
в оригинале: аксиомы там — контракт. Если реализация нарушает аксиому,
она считает уже не значение Шепли.

**B. Bilodeau, N. Jaques, P. W. Koh, B. Kim. «Impossibility theorems for
feature attribution».** PNAS 121(2), 2024.
[doi:10.1073/pnas.2304406120](https://doi.org/10.1073/pnas.2304406120),
[arXiv:2212.11870](https://arxiv.org/abs/2212.11870).
Теорема 3.3: для полного и линейного метода атрибуции (SHAP, Integrated
Gradients) специфичность ≤ 1 − чувствительность, то есть при выводе
локального поведения модели он не лучше угадывания. Граница класса методов,
а не претензия к реализациям.

**S. Wachter, B. Mittelstadt, C. Russell. «Counterfactual explanations without
opening the black box: automated decisions and the GDPR».** 2017.
[arXiv:1711.00399](https://arxiv.org/abs/1711.00399).
Контрфактуальное объяснение в смысле XAI: ближайший вход с другим ответом.
Полезно, чтобы не путать его с контрфактуалом Пёрла (тот же случай,
другая причина).

## 2. Атрибуция для агентов

**M. Horovicz. «AgentSHAP: Interpreting LLM agent tool importance with Monte
Carlo Shapley value estimation».** Препринт, 14.12.2025.
[arXiv:2512.12597](https://arxiv.org/abs/2512.12597).
Метод, который разбирается в ноутбуке 02: значение Шепли по инструментам
агента, $v(S)$ — близость ответа с набором $S$ к ответу с полным набором.
Эксперименты на 8 инструментах API-Bank, GPT-4o-mini. Код внутри
репозитория [GenAISHAP/TokenSHAP](https://github.com/GenAISHAP/TokenSHAP).
Статья называет себя «first framework for explaining tool importance in LLM
agents», но ShapleyFlow (ниже) вышел на десять месяцев раньше.

**M. Horovicz, R. Goldshmidt. «TokenSHAP: Interpreting large language models
with Monte Carlo Shapley value estimation».** NLP4Science Workshop, 2024,
с. 1–8. [doi:10.18653/v1/2024.nlp4science-1.1](https://doi.org/10.18653/v1/2024.nlp4science-1.1).
Предшественник AgentSHAP: то же самое по токенам промпта.

**Y. Yang, B. Huang, S. Qi et al. «Understanding and optimizing agentic
workflows via Shapley value»** (ShapleyFlow). Препринт, 01.02.2025.
[arXiv:2502.00510](https://arxiv.org/abs/2502.00510).
Значение Шепли по компонентам агентного workflow.

**C. Liu, Y. Zhang, Y. Zhong et al. «SkillShapley: Boundary-adaptive Shapley
valuation for skill step attribution in LLM agents».** Препринт, 13.08.2026.
[arXiv:2608.13173](https://arxiv.org/abs/2608.13173).
Атрибуция по шагам-скиллам вместо инструментов, двухфазный адаптивный
сэмплинг коалиций. Эксперименты на трёх задачах SkillsBench (бенчмарк
с детерминированными верификаторами). Бейзлайны двух видов: атрибуционные
(Individual, Leave-One-Out, Random Removal, LeastCore, точный Шепли)
и аппроксимации Шепли (Monte Carlo, Quasi-MC, парный MC, усечённый по
размеру коалиции). Ссылки на код в статье нет. AgentSHAP не цитирует,
TokenSHAP цитирует.

**B. Cohen-Wang, H. Shah, K. Georgiev, A. Mądry. «ContextCite: attributing
model generation to context».** NeurIPS 2024.
[arXiv:2409.00729](https://arxiv.org/abs/2409.00729),
код [MadryLab/context-cite](https://github.com/MadryLab/context-cite).
Атрибуция ответа к предложениям контекста через случайные абляции
и линейный суррогат. Нужны logprobs.

**F. Liu, N. Kandpal, C. Raffel. «AttriBoT: a bag of tricks for efficiently
approximating leave-one-out context attribution».**
[arXiv:2411.15102](https://arxiv.org/abs/2411.15102),
код [r-three/AttriBoT](https://github.com/r-three/AttriBoT).
Быстрое приближение leave-one-out, то есть ближе к необходимости. Нужны logprobs.

**P. R. Aryan, F. J. Ekaputra. «Semantic provenance and policy-aware
explanations for agentic coding systems».** Воркшоп TAAPAAI на ESWC 2026,
CEUR-WS vol. 4210, paper 6. [PDF](https://ceur-ws.org/Vol-4210/paper6.pdf).
Пока единственная работа, цитирующая AgentSHAP: упоминает его в ряду
методов атрибуции и предлагает вместо этого провенанс по трассам
исполнения. Связка с провенансом из второго акта лекции.

*Фон: где рядом с атрибуцией лежат сжатие промптов и чувствительность к формулировке.*

- H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, L. Qiu. «LLMLingua: compressing prompts
  for accelerated inference of large language models». EMNLP 2023, с. 13358–13376.
  [doi:10.18653/v1/2023.emnlp-main.825](https://doi.org/10.18653/v1/2023.emnlp-main.825).
  И H. Jiang et al. «LongLLMLingua». ACL 2024, с. 1658–1677.
  [doi:10.18653/v1/2024.acl-long.91](https://doi.org/10.18653/v1/2024.acl-long.91).
  Не атрибуция: важность токена по перплексии малой модели, для сжатия.
  Код и пакет [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua).
- E. Mosca, F. Szigeti, S. Tragianni, D. Gallagher, G. Groh. «SHAP-based
  explanation methods: a review for NLP interpretability». COLING 2022,
  с. 4593–4603. [ACL Anthology 2022.coling-1.406](https://aclanthology.org/2022.coling-1.406/).
  Обзор SHAP-вариантов для текста.
- Z. Jiang, F. F. Xu, J. Araki, G. Neubig. «How can we know what language
  models know?». TACL 8, 423–438, 2020.
  [doi:10.1162/tacl_a_00324](https://doi.org/10.1162/tacl_a_00324).
  Ответ модели сильно зависит от формулировки промпта.

## 3. Почему абляция врёт

**T. McGrath, M. Rahtz, J. Kramár, V. Mikulik, S. Legg. «The Hydra effect:
emergent self-repair in language model computations».** Препринт, 2023.
[arXiv:2307.15771](https://arxiv.org/abs/2307.15771).
Если убрать слой внимания, другие слои частично компенсируют его вклад.
Измеренный эффект абляции занижает важность компонента.

**C. Rushing, N. Nanda. «Explorations of self-repair in language models».**
ICML 2024. [arXiv:2402.15390](https://arxiv.org/abs/2402.15390).
Самовосстановление есть у разных семейств и размеров моделей; частично оно
объясняется финальным LayerNorm и разреженными наборами нейронов.

**D. Janzing, L. Minorics, P. Blöbaum. «Feature relevance quantification in
explainable AI: a causal problem».** AISTATS 2020, PMLR 108:2907–2916.
[arXiv:1910.13413](https://arxiv.org/abs/1910.13413).
Чем заменять «убранный» признак: интервенционная постановка против
обсервационной.

**S. Hooker, D. Erhan, P.-J. Kindermans, B. Kim. «A benchmark for
interpretability methods in deep neural networks»** (ROAR). NeurIPS 2019.
[arXiv:1806.10758](https://arxiv.org/abs/1806.10758).
Качество атрибуции через удаление признаков с переобучением модели.
На агента не переносится: его не переобучить.

**Y. Rong, T. Leemann, V. Borisov, G. Kasneci, E. Kasneci. «A consistent and
efficient evaluation strategy for attribution methods»** (ROAD). ICML 2022.
[arXiv:2202.00449](https://arxiv.org/abs/2202.00449).
То же без переобучения; устраняет утечку информации через форму маски.

**S. Krishna, T. Han, A. Gu et al. «The disagreement problem in explainable
machine learning: a practitioner's perspective».** TMLR, 2024.
[arXiv:2202.01602](https://arxiv.org/abs/2202.01602).
Разные методы атрибуции дают разные ответы: на COMPAS 14 из 15 пар методов
имеют отрицательную ранговую корреляцию. 86 % опрошенных практиков
разрешают расхождения эвристиками или не знают, как.

## 4. Объяснения, которым нельзя верить

**M. Turpin, J. Michael, E. Perez, S. R. Bowman. «Language models don't always
say what they think: unfaithful explanations in chain-of-thought prompting».**
NeurIPS 2023. [arXiv:2305.04388](https://arxiv.org/abs/2305.04388).
Подсказка в промпте смещает ответ (точность падает до 36 %), а в объяснении
о ней ни слова.

**T. Lanham, A. Chen, A. Radhakrishnan et al. «Measuring faithfulness in
chain-of-thought reasoning».** Препринт, 2023.
[arXiv:2307.13702](https://arxiv.org/abs/2307.13702).
В 7 из 8 задач модель на 175B опирается на свой CoT меньше, чем модель на 13B;
самое верное рассуждение в 6 из 8 задач — у 13B. Зависимость от размера
немонотонна.

**C.-E. Sun, L. Liu, G. Yan, Z. Wang, T.-W. Weng. «LLM agents already know when
to call tools — even without reasoning».** Препринт, 2026.
[arXiv:2605.09252](https://arxiv.org/abs/2605.09252).
Нужен ли инструмент, линейно декодируется из представления модели до начала
генерации: AUROC 0.89–0.96 на шести моделях. Решение принято до того,
как написано объяснение.

**A. P. Gema, N. Rajani, R. Saxena, W.-C. Kwan, P. Minervini.
«Chain-of-thought faithfulness of reasoning models varies with where and how
preference cues are delivered».** Препринт, 2026.
[arXiv:2608.29464](https://arxiv.org/abs/2608.29464).
Подсказку, пришедшую через ответ инструмента, модели проговаривают в CoT
реже, чем подсказку из сообщения пользователя, — у всех 15 проверенных
открытых моделей. Прямо про агентов: канал инструмента объясняется хуже всего.

**Y. Gur-Arieh, A. Marasović, M. Geva. «Faithfulness metrics don't measure
faithfulness: a meta-evaluation with ground truth».** Препринт, 2026.
[arXiv:2605.25052](https://arxiv.org/abs/2605.25052).
Бенчмарк BonaFide: большинство метрик верности CoT различают верные и
неверные рассуждения на уровне AUROC ≈ 0.5. Лучшие — CC-SHAP (0.70)
на уровне всего CoT и Filler Tokens (0.59) на уровне шага. Читать перед тем,
как брать любую метрику верности.

## 5. Бенчмарки траекторий и атрибуция отказов

Четыре датасета, которые лежат в `data/trajectories/` и используются
в ноутбуке 04 (что изменено при копировании — в `data/README.md`):

**J. Chen, Y. Sun, L. Zhang, L. Xu, J. Shi. «Long-horizon agent trajectory
attribution: a unified benchmark and fine-grained annotation framework».**
Препринт, 07.08.2026. [arXiv:2608.06909](https://arxiv.org/abs/2608.06909),
данные [chenjing-2024/agent-trajectory-attribution](https://github.com/chenjing-2024/agent-trajectory-attribution).
1 351 траектория из AgentDojo и Agent3Sigma: целевое действие, шаг-причина,
цепочки атаки и исполнения. Разметка сделана LLM и правилами, не людьми.

**HINTBench** — внутренние риски в длинных траекториях без атак.
[Jiacheng-31/HINTBench](https://github.com/Jiacheng-31/HINTBench),
Apache-2.0; ссылки на статью в репозитории нет.

**S. Zhang, M. Yin, J. Zhang et al. «Which agent causes task failures and when?
On automated failure attribution of LLM multi-agent systems»** (Who&When).
ICML 2025. [arXiv:2505.00212](https://arxiv.org/abs/2505.00212).
184 провала мультиагентных систем, размечены виновный агент и решающий шаг.
Лучший метод по агенту (All-at-once) даёт 53.5 %, лучший по шагу
(Step-by-step) — 14.2 % (GPT-4o, в среднем по настройкам).

**M. Chen, J. Wang, F. Mu et al. «Seeing the whole elephant: a benchmark for
failure attribution in LLM-based multi-agent systems»** (TraceElephant).
ACL 2026. [arXiv:2604.22708](https://arxiv.org/abs/2604.22708).
220 полностью наблюдаемых трасс и запускаемый код систем, то есть можно
делать контрфактуальные повторы. Согласие разметчиков: α Криппендорфа
0.72 по агенту, 0.64 по шагу.

Ещё про атрибуцию отказов:

- **M. Cemri, M. Z. Pan, S. Yang et al. «Why do multi-agent LLM systems fail?»**
  (MAST). NeurIPS 2025, Datasets & Benchmarks.
  [arXiv:2503.13657](https://arxiv.org/abs/2503.13657).
  Таксономия из 14 режимов отказа, 1 642 трассы из 7 фреймворков;
  согласие людей при построении таксономии κ = 0.88.
- **D. Deshpande, V. Gangal, H. Mehta et al. «TRAIL: trace reasoning and agentic
  issue localization».** Препринт, 2025.
  [arXiv:2505.08638](https://arxiv.org/abs/2505.08638).
  148 трасс, 841 размеченная ошибка; лучшая модель (Gemini-2.5-Pro) — 11 %.
- **G. Zhang, J. Wang, J. Chen et al. «AgenTracer: who is inducing failure in
  the LLM agentic systems?».** Препринт, 2025.
  [arXiv:2509.03312](https://arxiv.org/abs/2509.03312).
- **F. Kong, R. Zhang, H. Yin et al. «Aegis: automated error generation and
  attribution for multi-agent systems».** ICLR 2026.
  [arXiv:2509.14295](https://arxiv.org/abs/2509.14295). 9 533 траектории.
- **J. Liu, H. Xi, S. Zhang et al. «Who&When Pro: can LLMs really attribute
  failures in AI agents?».** Препринт, 10.07.2026.
  [arXiv:2607.09996](https://arxiv.org/abs/2607.09996).
  12 326 провальных трасс с эталоном из 26 бенчмарков.
- **Y. Li, H. Luo, Y. Xie et al. «ATBench: a diverse and realistic agent
  trajectory benchmark for safety evaluation and diagnosis».** COLM 2026.
  [arXiv:2604.02022](https://arxiv.org/abs/2604.02022).
  Безопасность траектории целиком, без меток шага.
- **J. Kim, S. Mun, S. Lee et al. «Towards faithful agentic XAI: a verification
  method and an open-world benchmark for better model faithfulness»**
  (CRAFTER-XAI-Bench). Препринт, 2026.
  [arXiv:2605.27879](https://arxiv.org/abs/2605.27879).
  Проверяет агентов-объяснителей на верность; SHAP там инструмент агента,
  а не сравниваемый метод.

## 6. Безопасность агентов: что меряют и чем

- **E. Debenedetti, J. Zhang, M. Balunović et al. «AgentDojo: a dynamic
  environment to evaluate prompt injection attacks and defenses for LLM
  agents».** NeurIPS 2024, Datasets & Benchmarks.
  [arXiv:2406.13352](https://arxiv.org/abs/2406.13352).
- **Q. Zhan, Z. Liang, Z. Ying, D. Kang. «InjecAgent: benchmarking indirect
  prompt injections in tool-integrated LLM agents».** Findings of ACL 2024.
  [arXiv:2403.02691](https://arxiv.org/abs/2403.02691).
- **H. Zhang, J. Huang, K. Mei et al. «Agent Security Bench (ASB)».** ICLR 2025.
  [arXiv:2410.02644](https://arxiv.org/abs/2410.02644).
- **Agent3Sigma** — трёхуровневая оценка безопасности агентов (статика,
  симуляция, реальные инструменты). [antgroup/Agent3Sigma](https://github.com/antgroup/Agent3Sigma).

Числа между этими бенчмарками напрямую не сопоставимы: у каждого своя
постановка.

**M. Dziemian, M. Lin, X. Fu et al. «How vulnerable are AI agents to indirect
prompt injections? Insights from a large-scale public competition».**
Препринт, 2026. [arXiv:2603.15714](https://arxiv.org/abs/2603.15714).
41 сценарий, 13 моделей; доля успешных атак от 0.5 % до 8.5 % в зависимости
от модели — разброс больше чем на порядок. Главный аргумент за измерение
вместо интуиции.

**A. Zou, M. Lin, E. Jones et al. «Security challenges in AI agent deployment:
insights from a large scale public competition».** Препринт, 2025.
[arXiv:2507.20526](https://arxiv.org/abs/2507.20526).
1.8 млн атак на 22 агента в 44 сценариях, больше 60 000 успешных.

**M. Nasr, N. Carlini, C. Sitawarin et al. «The attacker moves second:
stronger adaptive attacks bypass defenses against LLM jailbreaks and prompt
injections».** Препринт, 2025. [arXiv:2510.09023](https://arxiv.org/abs/2510.09023).
Адаптивные атаки обошли 12 недавних защит, у большинства с успехом выше 90 %,
хотя большинство этих защит заявляли почти нулевой успех атак. Прочитать
перед тем, как поверить любой цифре защиты.

## 7. Провенанс

**D. E. Denning. «A lattice model of secure information flow».**
Communications of the ACM 19(5), 236–243, 1976.
[doi:10.1145/360051.360056](https://doi.org/10.1145/360051.360056).
Откуда растут метки и их распространение: решётка уровней безопасности.

**Taint mode в Perl** — [perlsec](https://perldoc.perl.org/perlsec).
Самый наглядный пример: данные извне помечены, и если помеченное доходит
до `exec` без проверки, программа падает.

## 8. Как валидируют интерпретируемость вообще

Карта поля, если захочется копать за пределы атрибуции.

| Семейство | Канонические работы | Известная критика |
|---|---|---|
| Feature attribution | ERASER (DeYoung et al., ACL 2020, [arXiv:1911.03429](https://arxiv.org/abs/1911.03429)); ROAR; ROAD | удаление уводит вход из распределения |
| Sanity checks | Adebayo et al., «Sanity checks for saliency maps», NeurIPS 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292) | часть методов не отличает обученную сеть от случайной |
| Attention как объяснение | Jain & Wallace, NAACL 2019, [arXiv:1902.10186](https://arxiv.org/abs/1902.10186); Wiegreffe & Pinter, EMNLP 2019, [arXiv:1908.04626](https://arxiv.org/abs/1908.04626) | Bastings & Filippova, 2020, [arXiv:2010.05607](https://arxiv.org/abs/2010.05607): спор не решить без определения объяснения |
| Probing | Hewitt & Liang, control tasks, EMNLP 2019, [arXiv:1909.03368](https://arxiv.org/abs/1909.03368); Elazar et al., amnesic probing, TACL 2021, [arXiv:2006.00995](https://arxiv.org/abs/2006.00995) | точность пробы не равна использованию признака |
| Activation patching | Zhang & Nanda, ICLR 2024, [arXiv:2309.16042](https://arxiv.org/abs/2309.16042) | чувствительность к выбору испорченного распределения |
| Circuits | IOI (Wang et al., [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)); ACDC (Conmy et al., NeurIPS 2023, [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)); MIB (Mueller et al., ICML 2025, [arXiv:2504.13151](https://arxiv.org/abs/2504.13151)) | много ручной работы, плохо масштабируется |
| SAE | SAEBench (Karvonen et al., ICML 2025, [arXiv:2503.09532](https://arxiv.org/abs/2503.09532)); SynthSAEBench ([arXiv:2602.14687](https://arxiv.org/abs/2602.14687)) | feature absorption; прокси-метрики не переносятся |
| Concept-based | TCAV (Kim et al., ICML 2018, [arXiv:1711.11279](https://arxiv.org/abs/1711.11279)) | утечка концептов между слоями |
| Counterfactuals | CARLA (Pawelczyk et al., NeurIPS 2021, [arXiv:2108.00783](https://arxiv.org/abs/2108.00783)) | объяснение можно подогнать |
| CoT faithfulness | BonaFide (раздел 4) | большинство метрик на уровне AUROC ≈ 0.5 |
| Уровень агента | CRAFTER-XAI-Bench, TraceElephant (раздел 5) | семейство только складывается |

Пять режимов отказа, которые повторяются во всех семействах:

1. Вмешательство уводит вход из распределения.
2. Валидация круговая: метрика совпадает с тем, что метод оптимизирует.
3. Ground truth нет — нет разметки «вот что на самом деле повлияло».
4. Метрику можно обмануть: объяснение подгоняется без изменения модели.
5. Механизм компенсируется: удалённый компонент замещается другим.

Знать эти пять полезнее, чем знать любой отдельный метод. Всё равно все не выучить = ) 
