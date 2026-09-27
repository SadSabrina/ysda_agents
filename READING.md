# Что почитать

Сгруппировано по блокам лекции. Возле каждой работы — зачем именно она,
а не пересказ аннотации.

Точные arXiv-номера проставлены только там, где я в них уверен. В остальных
случаях — автор, название и год: ищется за десять секунд, а выдуманный
идентификатор хуже отсутствующего.

---

## Достаточность против необходимости

**Judea Pearl, «Causality»** (гл. 9, probabilities of causation) — откуда
берутся PN и PS. Если читать одно место из всей книги, то это оно: там
показано, что необходимость и достаточность независимы, и приведены границы,
внутри которых они вообще идентифицируемы из наблюдений.

**Watson, Gultchin, Taly, Floridi, «Local explanations via necessity and
sufficiency: unifying theory and practice»** (2021) — Proposition 1: value
function, над которой считают значения Шепли в стандартной постановке,
тождественно равна вероятности достаточности. Это формальное основание всей
критики A. Читать до разбора любого SHAP-подобного метода.

**Bilodeau, Jaques, Koh, Kim, «Impossibility theorems for feature
attribution»** (PNAS, 2024) — для любого метода, удовлетворяющего полноте
и линейности, специфичность ≤ 1 − чувствительность. Граница класса методов,
а не претензия к реализациям.

**Shapley, «A value for n-person games»** (1953) — четыре аксиомы и
единственность. Короткая, и её стоит прочитать в оригинале: аксиомы там
подаются как контракт, что и есть главная мысль лекции.

## Атрибуция для агентов

Все ссылки в этом разделе проверены 27.09.2026.

**Miriam Horovicz, «AgentSHAP: Interpreting LLM Agent Tool Importance with
Monte Carlo Shapley Value Estimation»**, arXiv:2512.12597, 14 декабря 2025.
Fiverr Labs, единственный автор. Метод, который разбираем. Эксперименты на
API-Bank, 8 инструментов, Level-1 test set.
⚠️ Не путать с Roni Goldshmidt (Nexar) — он соавтор *TokenSHAP* и
со-мейнтейнер репозитория, но не автор AgentSHAP.

**Состояние воспроизведения — проверено.** За девять с половиной месяцев
Semantic Scholar показывает **ровно одно** цитирование, и то в related work.
Независимых реализаций нет; единственная — авторская. Репозиторий
GenAISHAP/TokenSHAP: ~89 звёзд, 12 форков, мейнтейнеры — сами авторы.
В кураторском списке awesome-auditable-ai Shapley-методы для агентов
не упомянуты вообще.

**Единственная цитирующая работа — и она про провенанс.**
Peb Ruswono Aryan, Fajar J. Ekaputra, «Semantic Provenance and Policy-Aware
Explanations for Agentic Coding Systems», воркшоп TAAPAAI на ESWC'26,
10 мая 2026. https://ceur-ws.org/Vol-4210/paper6.pdf
AgentSHAP там упомянут одной строкой в related work, в пассаже про то, что
XAI для tool-augmented LLM «primarily focused on attribution». Сама работа
предлагает causal/policy-linked provenance. То есть единственный, кто
сослался на AgentSHAP, сделал это чтобы от него оттолкнуться в сторону
провенанса. Прямое подтверждение лестницы из третьего акта.

**«SkillShapley: Boundary-Adaptive Shapley Valuation for Skill Step
Attribution in LLM Agents»**, arXiv:2608.13173, 13 августа 2026.
Chang Liu, Yuqi Zhang, Yiman Zhong, Boyi Liu, Hengjun Wang (Beihang),
Shuyue Wei (Shandong, corresponding). Атрибуция по шагам-скиллам вместо
инструментов; двухфазный адаптивный сэмплинг коалиций. Эксперименты на
SkillsBench, три задачи с детерминированными верификаторами. Бейзлайны:
Individual, LOO, Random Removal, LeastCore, Monte Carlo Shapley,
Quasi-MC, Paired MC, size-k-truncated.
**Код не опубликован** (не «кода нет» — репозиторий может появиться).
Ключевая деталь: SkillShapley **не цитирует AgentSHAP**, хотя цитирует
TokenSHAP. Две ближайшие работы по Shapley для агентов существуют
в изоляции друг от друга.

**«Understanding and Optimizing Agentic Workflows via Shapley value»**,
arXiv:2502.00510, февраль 2025. Предшествует AgentSHAP на десять месяцев.
⚠️ Важно: AgentSHAP заявляет себя как «first framework for explaining tool
importance in LLM agents». С учётом этой работы приоритет под вопросом —
не повторяйте формулировку «первый» со сцены.

**TokenSHAP**, github.com/GenAISHAP/TokenSHAP — код, который разбираем.
Смотреть `token_shap/base.py` (там захардкожена `temperature=0.5` и в
`generate()`, и в `generate_with_tools()`; там же `_get_df_per_combination`,
где видно, что векторизация идёт по всему корпусу разом) и
`token_shap/agent_shap.py`.

TODO: проверить про что каждая 
[16] Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. How can we know what language
models know? Transactions of the Association for Computational Linguistics, 8:423–438, 2020.
doi: 10.1162/tacl_a_00324.
[17] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pages 13358–13376, 2023.
doi: 10.18653/v1/2023.emnlp-main.817.
[18] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili
Qiu. LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt
12
compression. In Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics, pages 1658–1677, 2024. doi: 10.18653/v1/2024.acl-long.89.
[19] Edoardo Mosca, Ferenc Szigeti, Stella Tragianni, Daniel Gallagher, and Georg Groh. SHAPbased explanation methods: A review for NLP interpretability. In Proceedings of the 29th
International Conference on Computational Linguistics, pages 4593–4603, 2022. doi: 10.18653/
v1/2022.coling-1.393.
[20] Miriam Horovicz and Roni Goldshmidt. TokenSHAP: Interpreting large language models with
Monte Carlo Shapley value estimation. In Proceedings of the 1st Workshop on NLP for Science
(NLP4Science), pages 1–8, 2024. doi: 10.18653/v1/2024.emnlp-nlpscience.1.

## Бенчмарки атрибуции: что есть и чего нет

Раздел появился после проверки. Формулировку «бенчмарка не существует»
использовать нельзя — опровергается одной ссылкой из зала.

**Контрпример, который надо знать.** Jing Chen, Yang Sun, Li Zhang, Lin Xu,
Jie Shi (Huawei), «Long-Horizon Agent Trajectory Attribution: A Unified
Benchmark and Fine-Grained Annotation Framework», arXiv:2608.06909,
7 августа 2026. 1300+ размеченных траекторий из AgentDojo и Agent3Sigma,
ground truth по корневой причине и по причинным цепочкам, две задачи.
Авторы прямо заявляют цель — систематическое сравнение методов атрибуции.

Оговорки, все в вашу пользу:
- Артефакта фактически нет: в репозитории chenjing-2024/agent-trajectory-attribution
  два коммита и README, данные и код заявлены как forthcoming.
- Всего два бейзлайна (Incremental и Leave-One-Out), авторы сами называют их
  «intentionally simple reference methods».
- Shapley-методов там нет вообще.
- Препринт, рецензирование не подтверждено.

**Не контрпримеры, но знать надо.** Бенчмарков атрибуции *отказа* много,
и их легко спутать с атрибуцией *вклада*. Там ищут, какой агент или шаг
сломал задачу, а методы — стратегии LLM-судейства, а не feature attribution:
- Who&When, arXiv:2505.00212, ICML 2025
- TRAIL, arXiv:2505.08638, 148 траекторий, 841 размеченная ошибка
- AEGIS, arXiv:2509.14295, ICLR 2026, 9533 траектории
- TraceElephant, arXiv:2604.22708, ACL 2026 — сравнивает 5 методов против
  человеческой разметки, Krippendorff alpha 0.72 / 0.64
- Who&When Pro, arXiv:2607.09996, июль 2026 — 12 326 траекторий отказов,
  3 метода × 10 моделей против golden labels

**CRAFTER-XAI-Bench**, arXiv:2605.27879 — тоже не то: оценивает агентные
XAI-системы по faithfulness, а SHAP там используется как инструмент внутри
агента, а не как сравниваемый метод.

**Формулировка на слайд:**
> Бенчмарков атрибуции отказов агента уже много — Who&When, TRAIL, AEGIS,
> TraceElephant. А бенчмарка, сравнивающего методы атрибуции вклада
> инструментов и шагов, по сути нет: единственная заявка — препринт Huawei
> от августа 2026 (arXiv:2608.06909), где два простых бейзлайна,
> ни одного Shapley-метода и неопубликованные данные.

## Почему абляция врёт

**McGrath, Rahtz, Kramar, Mikulik, Legg, «The Hydra effect: emergent
self-repair in language model computations»** (2023) — при удалении
компонента сеть частично компенсирует его вклад другими путями. Следствие:
измеренный эффект абляции — нижняя оценка важности, а не оценка.

**Rushing, Nanda, «Explorations of self-repair in language models»** (2024) —
подробнее про механизм и про то, насколько он распространён.

**Janzing, Minorics, Blöbaum, «Feature relevance quantification in explainable
AI: a causal problem»** (2020) — интервенционная постановка против
обсервационной. Ответ на вопрос «чем вообще заменять удалённый признак».

**Hooker, Erhan, Kindermans, Kim, «A benchmark for interpretability methods
in deep neural networks»** (ROAR, 2019) и **Rong et al.** (ROAD, 2022) —
как меряют качество атрибуции и почему удаление уводит вход off-manifold.
ROAR на агентов не переносится: агента нельзя переобучить. ROAD переносится.

**Krishna et al., «The disagreement problem in explainable machine
learning»** (2022) — разные методы дают разные ответы, ранговая корреляция
между ними 0.04–0.52, и практики не знают, что с этим делать.

## Объяснения, которым нельзя верить

**Turpin, Michael, Perez, Bowman, «Language models don't always say what they
think»** (2023) — подсказанный ответ в промпте меняет решение, а в
объяснении об этом ни слова.

**Lanham et al., «Measuring faithfulness in chain-of-thought reasoning»**
(2023) — обратное масштабирование: чем больше модель, тем менее верны
её объяснения, на 7 задачах из 8.

**Sun et al.** — вызов инструмента декодируется из внутреннего представления
модели ещё до генерации, AUROC 0.89–0.96. То есть решение уже принято,
а текст объяснения пишется после. Это граница, за которой black-box
не разрешается в принципе.

**Gema et al.** — канал возврата инструмента вербализуется хуже прочих
во всех 15 протестированных моделях.

**BonaFide** — мета-оценка метрик верности CoT: большинство из них работают
на уровне AUROC ≈ 0.5. Читать перед тем, как брать любую метрику верности.

## Атрибуция отказов в агентных системах

Полный список с проверенными ссылками — выше, в разделе про бенчмарки.
Здесь только то, что стоит помнить числами:

**Who&When** — атрибуция отказа к агенту получается в 53,5 % случаев,
к конкретному шагу — в 14,2 %. «Найти виноватый шаг» решается плохо даже
людьми с полной трассой.

**MAST** — таксономия из 14 режимов отказа мультиагентных систем,
1642 размеченные трассы, согласие разметчиков κ = 0,88.

**TRAIL**, **AgenTracer** — та же задача, у SOTA-моделей меньше 10 %.

Это важно для третьего акта: если атрибуция отказа к одному шагу плохо
решается даже постфактум с полной трассой на руках, то записывать связь
по ходу (провенанс) — не педантизм, а единственный способ её получить.

## Безопасность агентов: что меряют и чем

**AgentDojo**, **InjecAgent**, **Agent Security Bench** — воспроизводимые
бенчмарки атак на агентов. Разные постановки, числа между ними
не сопоставимы напрямую.

**Gray Swan IPI Arena OS** — 41 сценарий; по их публичным данным
1.8 млн атак на 22 агента, более 60 000 успешных, ASR по агентам
0.5 %–8.5 %. Разброс на порядок между агентами — главный аргумент
за измерение вместо интуиции.

**«The Attacker Moves Second»** (октябрь 2025) — адаптивные атаки обошли
12 опубликованных защит, у большинства с успехом выше 90 %, при том что
авторы защит сообщали почти нулевой ASR. Обязательно к прочтению перед тем,
как поверить любой заявленной цифре защиты.

## Как валидируют интерпретируемость вообще

Карта поля, если захочется копать за пределы атрибуции.

| Семейство | Канонический бенчмарк | Известная критика |
|---|---|---|
| Feature attribution | ERASER, ROAR, ROAD | off-manifold при удалении |
| Sanity checks | Adebayo et al., randomization tests | часть методов не отличает обученную сеть от случайной |
| Attention как объяснение | Jain & Wallace против Wiegreffe & Pinter | Bastings & Filippova: спор нерешаем без определения объяснения |
| Probing | control tasks, MDL, amnesic probing | точность пробы не равна использованию признака |
| Activation patching | Zhang & Nanda; causal abstraction, IIA | чувствительность к выбору corrupted-распределения |
| Circuits | IOI, ACDC, MIB | ручная разметка, плохо масштабируется |
| SAE | SAEBench, SynthSAEBench | feature absorption; прокси-метрики не переносятся |
| Concept-based | TCAV | утечка концептов между слоями |
| Counterfactuals | CARLA | manipulability: объяснение подгоняется |
| CoT faithfulness | BonaFide | большинство метрик на AUROC ≈ 0.5 |
| Agent-level | CRAFTER-XAI-Bench, TraceElephant | семейство только формируется |

Пять режимов отказа, которые повторяются во всех семействах независимо:

1. Вмешательство уводит вход из распределения.
2. Валидация круговая: метрика совпадает с критерием оптимизации метода.
3. Ground truth отсутствует — нет разметки «вот что на самом деле повлияло».
4. Метрика геймится: объяснение подгоняется без изменения модели.
5. Механизм компенсируется: удалённый компонент замещается другим.

Знание этих пяти дороже знания любого отдельного метода.
