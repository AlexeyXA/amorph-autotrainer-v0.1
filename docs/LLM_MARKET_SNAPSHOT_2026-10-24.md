# AMORPH — снимок публичного рынка LLM: цены и benchmark evidence

**Дата снимка:** 24 октября 2026 г.  
**Файлы данных:** [CSV-каталог](../data/llm_market_snapshot_2026-10-24.csv).  
**Назначение:** база для проектирования cost-aware Selector; это не рейтинг «лучшей модели» и не доказательство экономии на реальной бизнес-задаче.

## Суть сбора

Собраны публичные API-тарифы и спецификации OpenAI, Google Gemini, xAI, Anthropic, DeepSeek и Mistral. Для ряда моделей добавлены внешние сигналы:
- **Artificial Analysis Intelligence Index v4.3.2** — сводная оценка по десяти evaluation-наборам, скорость и рассчитанная стоимость одной задачи в их benchmark.
- **Text Arena** — человеческие предпочтения в открытых диалогах, score с неопределённостью и число голосов.

Эти измерения не взаимозаменяемы и не заменяют тест на реальной целевой задаче. Цена токенов, AA index и Arena score нельзя объединять в один показатель без заранее заданной методики.

## Цены и публичные сигналы

Все ставки ниже — USD за 1 млн токенов. Проверка выполнена 2026-10-24; цены, алиасы и условия доступности могут измениться.

| Провайдер / вариант | Input | Cached input | Output | Контекст | AA Index / AA task cost |
|---|---:|---:|---:|---:|---|
| OpenAI GPT-6 Luna (Max) | $0.10 | $0.01 | $0.50 | 1M | 38 / $0.07 |
| OpenAI GPT-6.1 Sol (Max) | $2.00 | $0.10 | $10.00 | 1M | 52 / $0.72 |
| OpenAI GPT-6 Astra (Max) | $10.00 | $1.00 | $50.00 | 1M | 53 / $3.26 |
| Google Gemini 3.8 Flash (High) | $0.75* | $0.075* | $3.75* | 1M | 41 / $1.24 |
| Google Gemini 4 Argon (High) | **не подтверждена официально** | n/a | **не подтверждена официально** | 1M | 53 / $1.99 |
| xAI Grok 4.7 (Xhigh) | $2.00 | $0.50 | $6.00 | 500K | 46 / $3.74 |
| Anthropic Claude Haiku 5.5 (Max) | $0.10** | $0.01 | $0.50 | 1M | 43 / $0.21 |
| Anthropic Claude Sonnet 5.5 (Max with fallback) | $2.00 | $0.10 | $10.00 | 1M | 56 / $5.46 |
| Anthropic Claude Opus 5.5 (Max with fallback) | $4.00 | $0.20 | $20.00 | 1M | 58 / $5.98 |
| DeepSeek V4.1 Flash 1017 (Max) | $0.30*** | $0.006*** | $1.20*** | 1M | 45 / $0.26 |
| Mistral Small 4 | $0.15 | $0.015 | $0.60 | 256K | — |
| Mistral Large 3 | $0.50 | $0.05 | $1.50 | 256K | — |
| Mistral Medium 3.5 | $1.50 | $0.15 | $7.50 | 256K | — |

**Важно:** AA cost per task — цена выполнения задачи в составе Artificial Analysis eval-набора, рассчитанная по их фактическим токенам и тарифам. Это не цена любого пользовательского запроса и не стоимость успешного бизнес-результата.

- * Gemini 3.8 Flash: вводная цена $0.75/$3.75 действует до 31 декабря 2026 г.; с 1 января 2027 г. Google публикует $1.50/$7.50.
- ** Claude Haiku 5.5: входная ставка $0.10 для промптов до 100K токенов; выше нужен отдельный расчёт длинного контекста.
- *** DeepSeek API model ID — deepseek-flash. Указан peak-rate; off-peak input/cache/output — $0.15/$0.003/$0.60; peak — $0.30/$0.006/$1.20.
- Gemini 4 Argon: сторонние сравнительные страницы показывают $2/$10, но точную цену не удалось подтвердить в официальной таблице тарифов Gemini API, поэтому цена оставлена неизвестной и не участвует в денежном сценарии.

## Условный токен-сценарий

Для сравнения тарифов, но не как прогноз расходов пользователя, считаем 1 млн запросов, каждый из которых содержит 2 000 входных и 500 выходных токенов. Без кэша и только за input/output токены:

Формула: 2,000 × input price + 500 × output price (оба тарифа на 1 млн токенов).

| Модель | Цена 1 млн условных запросов |
|---|---:|
| GPT-6 Luna | $450 |
| Claude Haiku 5.5 | $450 |
| DeepSeek V4.1 Flash | $600 off-peak / $1,200 peak |
| Mistral Small 4 | $600 |
| Mistral Large 3 | $1,750 |
| Gemini 3.8 Flash | $3,375 до конца 2026 / $6,750 с 2027-01-01 |
| Grok 4.7 | $7,000 при prompt ниже 200K |
| GPT-6.1 Sol | $9,000 |
| Claude Sonnet 5.5 | $9,000 |
| Mistral Medium 3.5 | $6,750 |
| Claude Opus 5.5 | $18,000 |
| GPT-6 Astra | $45,000 |
| Gemini 4 Argon | не оценивать до подтверждения цены |

Не учтены инструменты/поиск, повторные вызовы, агентные циклы, ручная проверка, интеграция, хостинг и налоги. Для Grok 4.7 при prompt выше 200K rates вырастают до $4/$1/$12 за input/cached input/output; xAI указывает скидку 20% для Batch API. Для Gemini промо-тариф закончится в конце 2026 года.

## Что это говорит о задаче AMORPH

Сценарий показывает, что одни токен-ставки различаются на порядки. Но **минимальная цена токена не означает минимальную стоимость принятого результата**:
- дешёвый вариант может ошибаться чаще или не проходить Quality Gate;
- reasoning-модели могут производить разное количество токенов на одну задачу;
- повторные вызовы, инструменты, кэш и человеческая проверка способны изменить итог;
- некоторые benchmark-варианты (Max/High/Xhigh/fallback) не эквивалентны.

Поэтому будущий Selector должен сначала отсеивать кандидатов, которые не прошли критерии качества/безопасности, а затем считать полный TCO и стоимость успешного результата. Публичный leaderboard нужен для shortlist, но не для автоматической приёмки.

### Сигналы Text Arena

На более свежем снимке Text Arena за 22 октября 2026 г. указано 8,946,356 голосов и 418 моделей. Gemini 4 Argon (High) — 1517±7 при 10 083 голосах; Claude Opus 5.5 (High) — 1509±7 при 10 322 голосах; Gemini 3.8 Flash (High) — 1499±5 при 41 423 голосах; GPT-6.1 Sol (Max) — 1480±7 при 8 305 голосах. Эти score измеряют предпочтения в открытых диалогах, а не точность на закрытой бизнес-задаче. Сравнивать варианты разных reasoning-режимов нужно осторожно. Это рейтинг с изменяющимися оценками, поэтому дата среза сохраняется.

## Интеграционные правила для AMORPH

1. Хранить дату проверки, provider, закреплённый model ID/version, вариант reasoning, ставки кэша/длинного контекста и ссылки.
2. Считать расходы по нескольким token profiles (низкий/обычный/высокий выход), а не по одной выдуманной цене «за запрос».
3. Использовать публичные метрики только для shortlist; обязательный Quality Gate проводится на target-task evaluation.
4. Считать повторные попытки, инструменты, review и сопровождение. Целевая величина — цена успешного и принятого результата.
5. Неизвестные ставки помечать unknown, не заменять нулём и не включать в денежный optimum.
6. Обновлять снимок и проверять deprecation/alias. Не использовать latest без записи фактически разрешённой версии.
7. Не обучать модель на рейтинговых метаданных. Для обучения нужны task-specific примеры и корректный независимый split.

## Источники

**Официальные тарифы и спецификации**
- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing.md
- Gemini model catalog: https://ai.google.dev/gemini-api/docs/models
- xAI Grok 4.7: https://docs.x.ai/developers/grok-4-7
- xAI pricing: https://docs.x.ai/developers/pricing
- Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- DeepSeek pricing: https://api-docs.deepseek.com/quick_start/pricing/
- Mistral pricing: https://docs.mistral.ai/inference/pricing
- Mistral context limits: https://docs.mistral.ai/resources/known-limitations

**Внешние benchmark-источники**
- Artificial Analysis leaderboard: https://artificialanalysis.ai/leaderboards/models
- GPT-6 Luna vs Astra: https://artificialanalysis.ai/models/comparisons/gpt-6-luna-vs-gpt-6-astra
- GPT-6.1 Sol: https://artificialanalysis.ai/models/gpt-6-1-sol
- GPT-6 Astra: https://artificialanalysis.ai/models/gpt-6-astra
- Gemini 3.8 Flash: https://artificialanalysis.ai/models/gemini-3-8-flash
- Gemini 4 Argon: https://artificialanalysis.ai/models/gemini-4-argon
- Grok 4.7: https://artificialanalysis.ai/models/grok-4-7
- Claude Haiku 5.5: https://artificialanalysis.ai/models/claude-haiku-5-5
- Claude Sonnet 5.5: https://artificialanalysis.ai/models/claude-sonnet-5-5
- DeepSeek V4.1 Flash 1017: https://artificialanalysis.ai/models/deepseek-v4-1-flash-1017
- Text Arena: https://arena.ai/leaderboard/chat/text
