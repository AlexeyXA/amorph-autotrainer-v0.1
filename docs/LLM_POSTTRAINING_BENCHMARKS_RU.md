# Датасеты и бенчмарки для AMORPH: SFT, DPO, переобучение и самоулучшение

**Проверено:** 23 октября 2026 года. Документ разделяет уже проведённые тесты и будущие эксперименты; наличие ссылки на датасет не означает, что AMORPH уже умеет его использовать.

## 1. Текущее ограничение проекта

AMORPH AutoTrainer v0.1 — MVP классификации и извлечения признаков: TF-IDF, optional multilingual-E5-small и экспериментальный гибрид. В коде пока нет фактического обучения генеративной LLM через SFT/LoRA/QLoRA, preference tuning/DPO, reward model или GRPO. Поэтому v0.1 ещё не дообучает веса ChatGPT/Grok/Gemini и не является системой автономного самоулучшения LLM.

Текущий проект можно проверять на совместимых задачах и датасетах. Для обучения чат-моделей нужен отдельный backend, реальные запуски обучения, сохранённые адаптеры и отдельный протокол проверки переобучения.

## 2. Датасеты для SFT / instruction tuning

| Датасет | Содержание и размер | Лицензия / ограничения | Предлагаемая роль |
|---|---|---|---|
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Человеческие диалоги с человеческими оценками; 35 языков, включая русский (в карточке указано около 9 тыс. русских сообщений) | Карточка указывает Apache-2.0. Это диалоговые деревья: разбивать train/validation нужно по дереву разговора, не по отдельным сообщениям | **Предпочтительный первый русский SFT-пилот** после фиксации ревизии и аудита данных |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Более 15 тыс. созданных людьми instruction/response примеров | CC BY-SA 3.0; нужна атрибуция и соблюдение ShareAlike. Англоязычный набор | Малый англоязычный smoke test SFT |
| [RU Dataset 1](https://huggingface.co/datasets/ru-dataset/ru-dataset-small) | Русский chat/SFT: программирование, математика, reasoning, инструкции; обновляемая коллекция | Карточка заявляет Apache-2.0, но набор пополняется; надо закрепить commit и проверить происхождение, точные/семантические дубликаты, качество и возможное пересечение с оценочными наборами | Второй русский набор после data audit, не единственная опора для научного вывода |
| [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | Около 939 тыс. SFT-примеров из многих источников | Смешанная совокупность. Карточка предупреждает, что поднаборы имеют разные лицензии, некоторые non-commercial. Нельзя считать всю смесь автоматически коммерчески разрешённой | Исследовательский набор после отбора подмножеств и проверки лицензий |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | Около 1 млн преимущественно синтетических/собранных instruction и chat примеров | Нужен аудит происхождения, лицензий, дублей и качества; большой размер не гарантирует полезность | Не первый эксперимент; использовать только после аудита |
| [Yandex Geo Reviews 2023](https://github.com/yandex/geo-reviews-dataset-2023) | 500 тыс. русских пользовательских отзывов с рейтингом 0–5, организацией и адресом | Репозиторий указывает MIT; рейтинг — шумная proxy-разметка, не ручная разметка тональности и не intent | Отдельный PR #22 — классификация rating buckets, не SFT и не бенчмарк чат-бота |

Не надо объединять все эти корпуса в один train и судить о качестве только по train loss. У них разные лицензии, языки, домены и типы ответов. Для воспроизводимости фиксировать конкретный commit/revision и хеш исходных данных.

## 3. Preference datasets для DPO / reward models

| Датасет | Содержание | Ограничения / роль |
|---|---|---|
| [UltraFeedback Binarized Preferences Cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | Около 60,9 тыс. пар chosen/rejected, preference ratings; карточка указывает MIT | Английский. Оценки partly model-derived и могут иметь judge/style bias. Кандидат для отдельного DPO-пилота |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Около 169 тыс. записей с chosen/rejected предпочтениями helpfulness/harmlessness, а также red-team данные | Карточка указывает MIT, но прямо предупреждает, что исходные данные не предназначались для обычного SFT диалогового агента. Использовать для preference/reward исследования в рамках описанных ограничений, не как простой диалоговый train set |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Диалоговые деревья и оценки качества от людей | Возможный дополнительный сигнал предпочтений после отдельной схемы подготовки и проверки разметки |

DPO — отдельный objective и отдельный training backend, не ещё один классификатор текущего v0.1. До его реализации требуются фиксированный формат пар, baseline, loss logs, validation, независимый benchmark, ресурсы и сохранённый adapter artifact.

## 4. Бенчмарки для измерения качества LLM после дообучения

| Бенчмарк / инструмент | Что измеряет | Как применять |
|---|---|---|
| [MERA](https://mera.a-ai.ru/en/catalog): MERA Text / Industrial MERA | Русский язык, знания, рассуждение; Industrial MERA ориентирован на реальные бизнес-задачи | Главный внешний ориентир для будущей русскоязычной модели. Доступность данных и допустимые способы оценки нужно проверять отдельно для каждой задачи MERA |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | Проверяемое следование инструкциям и ограничениям формата | Считать выполнение каждого условия; хороший приёмочный тест для поведения чат-модели |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Текстовые математические задачи; карточка указывает MIT | Объективный тест reasoning. Не обучать на official test и не подбирать параметры по нему |
| [BBH](https://arxiv.org/abs/2210.09261) / [Open LLM Leaderboard tasks](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about) | Сложное логическое рассуждение и широкий набор навыков | Проверка переноса; разные task scores нельзя сливать в одно значение «интеллекта» без явного протокола |
| [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Склонность к распространённым заблуждениям и неистинным ответам | Внешний диагностический тест с явно зафиксированной версией scoring |
| [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Единый runner для множества задач, включая GSM8K, BBH, HumanEval и другие | Фиксировать версию runner, task IDs, prompts, few-shot/zero-shot и decoding параметры |
| [HumanEval](https://github.com/openai/human-eval) / MBPP | Исполняемая корректность сгенерированного кода | Дополнительный тест для code-моделей; запускать код только в изолированном sandbox |
| MT-Bench / AlpacaEval | Диалоговое качество через LLM-as-a-judge | Вспомогательный сигнал: фиксировать судью и prompt, учитывать position/style bias и дополнять объективными тестами |

Macro-F1 классификатора, математический exact match, instruction-following и субъективная оценка диалога измеряют разные свойства. Хороший результат на одном наборе не означает общего превосходства над ChatGPT, Grok или Gemini.

## 5. Как обнаруживать переобучение, забывание и ложное самоулучшение

Минимальная схема для обучения LLM:

- **Train** — только эти данные меняют веса.
- **Validation** — выбор эпохи, параметров и кандидата.
- **Acceptance gate** — заранее определённый парный критерий приёмки, не бесконечная площадка для настройки.
- **Untouched final test** — открывается после фиксации кандидата. После просмотра и следующего изменения модели этот test больше не считается нетронутым; нужен новый holdout или новая ревизия.
- **External regression suite** — навыки, языки и задачи, на которых модель не обучалась.

Обязательные измерения:

1. Training loss и validation loss по шагам/эпохам. Снижение train loss при росте validation loss — предупреждение об overfitting, но не единственный критерий.
2. До/после: одна и та же модель, версия tokenizer/chat template, одинаковые prompts и decoding settings.
3. Regression/catastrophic forgetting: не скрывать деградацию других навыков за улучшением одной целевой метрики.
4. Exact и near-duplicate audit между train/validation/gate/test; группировать по разговорному дереву, пользователю, документу, источнику или организации при наличии ID.
5. Повторить несколько seed/splits, если позволяет бюджет; показывать среднее и разброс.
6. Learning curve: сравнить, например, 100 / 500 / 2 000 train-примеров и 1 / 2 / 3 эпохи, а не только единственный удачный запуск.
7. Memorization audit с безопасными синтетическими canary-фразами; не добавлять в обучающие данные реальные секреты или личные данные ради проверки.
8. Денежный TCO: измеренные затраты на подготовку/разметку, обучение, CPU/GPU, память, инференс, expected traffic и обслуживание. Нулевой тариф в конфигурации означает допущение о бесплатном ресурсе, а не доказательство нулевой экономической стоимости.

Proof Gate должен уметь выдать REJECT или INCONCLUSIVE. Отсутствие доказанного ухудшения не является доказательством улучшения. Модель не может быть единственным судьёй своих синтетических ответов, а self-training данные, созданные самой моделью, не являются независимой разметкой автоматически.

## 6. Как правильно проверять закрытые и открытые чат-системы

- Open-weight модель можно дообучать через совместимый framework, например [Hugging Face TRL](https://huggingface.co/docs/trl/en/index) с SFT/PEFT; модель, лицензию, ревизию и ресурсы нужно закрепить заранее.
- Закрытым ChatGPT/Grok/Gemini API нельзя автоматически управлять весами. Можно сравнивать их ответы black-box eval на одинаковых заданиях; настройка возможна только через поддерживаемый поставщиком tuning path с доступом в конкретном аккаунте.
- Google указывает tuning для отдельных моделей через Vertex AI / Gemini Enterprise Agent Platform; прямой fine-tuning через Gemini API/AI Studio недоступен согласно их публичной документации на момент проверки. Актуальные модели и условия нужно сверить перед запуском.
- Страница цен OpenAI API в октябре 2026 сообщает о сворачивании fine-tuning platform для новых пользователей; доступность проверять в своём аккаунте, не считать интеграцию гарантированной.
- Для публичного пути обучения пользовательских весов Grok/xAI мне не удалось подтвердить актуальную стабильную публичную документацию в ходе этого поиска. Статус — **не подтверждено**, не утверждение, что такой функции точно нигде не существует.
- Поэтому для закрытых систем AMORPH может сначала сравнивать ответы и стоимость. Для реального обучения весов следует начинать с небольшой open-weight модели и отдельного backend.

## 7. Проверки AMORPH, которые уже фактически выполнены

- **BANKING77 / CLINC150:** оба внешних intent-бенчмарка прошли через GitHub Actions; ZIP digest и внутренние SHA-256 inventory проверены. На CLINC150 E5 прошёл acceptance gate; на BANKING77 gate выдал INCONCLUSIVE / SELF_COMPARISON, потому что выбранный baseline сравнивался с идентичными предсказаниями.
- **MASSIVE ru-RU:** TF-IDF/E5 и предварительно заданная конфигурация E5 C=0.1, class_weight=balanced выполнены. На этой конфигурации E5 дал заметный прирост и прошёл gate. Ограничения: задача voice-assistant intents, не B2B CRM; near-duplicate audit пропущен выше лимита; официальный test уже просматривался и не должен считаться нетронутым holdout.
- **Yandex Geo Reviews 2023:** отдельный PR #22 запускает классификацию rating buckets с group split по организации. Это реальные русские отзывы с proxy-label по рейтингу, не SFT и не intent benchmark. Unit tests прошли; итоговые метрики нужно брать из завершённого Actions artifact, а не предполагать до завершения.
- **TCO:** текущий README явно описывает selector как runtime proxy (fit time + ожидаемые запросы × latency). Несмотря на наличие cost estimator, v0.1 ещё не оптимизирует денежный TCO как целевую функцию.
- **LLM SFT/LoRA/QLoRA/DPO/GRPO:** не реализованы и не считаются пройденными экспериментами до появления фактического training backend и артефактов.

## 8. Рекомендуемая последовательность

**P0 — закрыть совместимые текущие тесты:** завершить PR #22, проверить provenance и результаты; не смешивать intent и rating proxy результаты.

**P1 — первый LLM SFT proof of concept:** небольшая open-weight модель и LoRA/PEFT; русский subset OASST1 с закреплённой ревизией и split по conversation tree; сравнить base model с адаптером на внешнем instruction-following benchmark и русских задачах, где доступ разрешён; записывать loss, время и артефакты. Провести ablation 100/500/все доступные train примеры, 1/2/3 эпохи и повторные seed.

**P2 — preference optimization:** отдельный DPO backend и пилот на UltraFeedback либо HH-RLHF с учётом назначения/лицензии источника.

**P3 — контролируемое самоулучшение:** генерация candidate examples → независимая проверка/человеческая выборка → дедупликация → новая pinned train revision → внешний regression benchmark. Без этого цикл может лишь масштабировать собственные ошибки.

## Источники

- [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [RU Dataset 1](https://huggingface.co/datasets/ru-dataset/ru-dataset-small)
- [Tülu 3 mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [MERA benchmark catalogue](https://mera.a-ai.ru/en/catalog)
- [Open LLM Leaderboard methodology](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about)
- [lm-evaluation-harness tasks](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md)
- [Hugging Face TRL](https://huggingface.co/docs/trl/en/index)
- [Gemini tuning](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/tuning)
- [OpenAI API pricing / fine-tuning notice](https://platform.openai.com/pricing/)
