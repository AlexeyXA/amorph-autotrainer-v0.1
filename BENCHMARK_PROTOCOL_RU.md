# AMORPH AutoTrainer — протокол проверки преимущества

## Цель

Проверить, достигает ли AMORPH заданного качества при меньшей полной стоимости, а не заявлять превосходство по одному бенчмарку.

## Наборы данных

Использовать минимум 6–10 реальных задач классификации/извлечения признаков из разных доменов. Для каждого набора зафиксировать лицензию, версию источника, схему меток, правила группировки и дату снимка. Synthetic demo не считается бенчмарком.

## Baselines

1. TF-IDF + Logistic Regression.
2. Предобученные multilingual embeddings + Logistic Regression.
3. Ручной выбор простейшего метода по заранее записанным правилам.
4. Для классического AutoML — FLAML/AutoGluon только на задачах, которые они действительно поддерживают.
5. Для будущего LLM fine-tuning — конкретный baseline соответствующего fine-tuning tool; не сравнивать инструменты на несовместимых задачах.

## Разделение данных

- Train: fit параметров модели.
- Validation: выбор стратегии и её гиперпараметров.
- Acceptance gate: один заранее определённый paired comparison с baseline, bootstrap interval и порогом качества.
- Final test: используется после фиксации выбора только для финального отчёта. После просмотра теста нельзя менять модель и продолжать считать тот же test независимым.

Для временных, пользовательских или source-group данных необходимо разделять по времени/группе, а не случайно. Near-duplicate similarity — только эвристика; пересечения следует дополнительно проверять доменными ключами.

## Метрики

- Классификация: Macro-F1 (основная), Balanced Accuracy, Accuracy, метрики каждого класса.
- Retrieval/embeddings при наличии размеченных пар: Recall@k, MRR или NDCG.
- Стоимость: полное время preprocessing/fit, задержка inference на репрезентативном batch, объём памяти, денежные ставки, ожидаемое число запросов, труд человека и обслуживание.
- Надёжность: регрессии на ранее принятом наборе, ошибки формата, ошибки обработки пустых/грязных входов.

## Критерий коммерческого преимущества

Для каждой задачи и заранее заданного рабочего объёма запросов сравнить минимальную стоимость подхода, который удовлетворил одним и тем же ограничениям качества/задержки. Публиковать среднее и разброс по задачам, случаи проигрыша, абсолютные затраты и версии всех артефактов. Нельзя делать вывод «универсально дешевле» по нескольким задачам или по синтетическим данным.

## Что означает статистический gate

Paired class-stratified bootstrap оценивает неопределённость разницы Macro-F1 на acceptance-gate выборке для уже обученных моделей. Он не оценивает вариативность при повторном обучении и не гарантирует работу на будущем распределении. Для более сильных выводов нужны независимые повторные split/seed и достаточный размер выборки.


## Gate protocol 1+3 (MASSIVE)
- Target fractions for train/validation/acceptance-gate/final-test: **0.45 / 0.15 / 0.25 / 0.15**.
- Before internal splitting, the benchmark may exclude whole training labels with fewer than 20 unique texts or fewer than four distinct worker groups after conflict removal and exact deduplication. Report counts and SHA-256 provenance for the removed rows; never filter the official test to accommodate train support.
- For a grouped split, the MILP must require at least five actual rows per retained label in acceptance gate and at least one row per label in train, validation, and internal test. It must count rows per group and label, not just group presence, and validate the returned assignment. If infeasible, fail closed.
- The eligibility thresholds are heuristic feasibility screens, not formal statistical power analysis. Gate sufficiency is a separate measured condition; official-test metrics remain report-only.
