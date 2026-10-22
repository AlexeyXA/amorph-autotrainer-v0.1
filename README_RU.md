# AMORPH AutoTrainer v0.1

Бюджетный воспроизводимый MVP для **классификации и извлечения признаков**. Версия не обучает базовую нейросеть с нуля. Основной маршрут использует TF-IDF + Logistic Regression; необязательный маршрут `e5` использует предобученный `intfloat/multilingual-e5-small` и линейный классификатор.

## Что уже входит

- CSV-приём данных с проверкой схемы и пустых строк.
- Консервативное обнаружение точных дубликатов и остановка при конфликтующих метках.
- Четырёхчастное разделение train/validation/acceptance-gate/final-test; целевые доли по умолчанию 45/15/25/15, группировка обнаруженных почти-дубликатов до 5 000 строк, необязательный split по `--group-column` и проверка пересечений между разделами.
- Strategy ABI / conformance checks, fingerprints и replay descriptor для аудита конфигурации и воспроизведения запуска.
- Дешёвый TF-IDF baseline.
- Необязательные мультиязычные эмбеддинги E5.
- Выбор кандидата по validation-метрике и времени обучения/инференса, без использования test при выборе.
- Итоговый paired class-stratified bootstrap интервал разницы Macro-F1 между выбранным кандидатом и baseline.
- Quality Gate с абсолютным порогом Macro-F1 (если указан) и проверкой non-inferiority.
- SQLite-журнал экспериментов, хеш исходного CSV, отчёт по данным и manifest модели с версиями Python/библиотек, Git commit (если доступен) и параметрами split.
- Фактические времена и оценка TCO по введённым ставкам. Стоимость — оценка, не счёт провайдера.
- Опциональная выгрузка признаков (разреженные TF-IDF матрицы или dense E5 embeddings).
- Жёсткий лимит времени и числа кандидатов.
- Демо-генератор, тесты и CLI для предсказаний/извлечения признаков.

## Ограничения v0.1 — честно

1. Встроены только два маршрута классификации: TF-IDF + Logistic Regression и необязательные multilingual-E5 эмбеддинги + Logistic Regression. RAG, LoRA/QLoRA, дистилляция и полное fine-tuning — будущие отдельные стратегии, а не объявленная поддержка.
2. Данные делятся на четыре непересекающиеся части: train, validation, acceptance gate и финальный test. Выбор стратегии использует validation; статистический gate использует отдельную acceptance-gate выборку; финальный test — только для отчёта и не меняет выбор модели.
3. Acceptance gate применяет paired class-stratified bootstrap для разницы Macro-F1 и требует минимум `--min-gate-examples-per-class` примеров каждого класса (по умолчанию 5). Для group-safe MILP-разбиения минимум фактических строк каждого класса в gate является жёстким ограничением и перепроверяется после решения; невозможность выполнения ограничений приводит к fail-closed. Для обычного stratified fallback ограничение не гарантируется самим splitter, поэтому статус gate отдельно проверяется pipeline. Минимум 5 — операционный предохранитель, а не формальная гарантия мощности теста. Bootstrap оценивает неопределённость gate-выборки при уже обученных моделях и не учитывает вариативность переобучения на новых train split.
4. Near-duplicate audit использует cosine similarity символьных n-грамм только до 5 000 строк. Это эвристика, не семантическое доказательство. В пределах лимита найденные кластеры не пересекают splits. Для больших наборов аудит явно отмечается как пропущенный. Без `--group-column` остаётся stratified split с защитой от точных дубликатов; при переданной колонке группы сохраняются вместе даже если near-duplicate audit пропущен. Для временной оценки нужен отдельный хронологический split: `--group-column` не заменяет split по времени. Не все смысловые дубликаты и лицензии обнаруживаются автоматически.
5. Runtime budget — best effort: система проверяет бюджет между стадиями и не запускает следующего кандидата после исчерпания лимита, но не может принудительно прервать уже выполняющийся вызов scikit-learn/encoder. В отчёте отдельно есть TCO выбранной модели и TCO полного процесса выбора стратегии; latency proxy измеряется для одиночного запроса на небольшом probe. Превышение отражается в отчёте и даёт код завершения 3.
6. Synthetic demo предназначен только для smoke test. Качество на искусственных текстах ничего не доказывает о реальной задаче.
7. Нулевая стоимость вычислений по умолчанию — предположение пользователя, а не факт об экономической стоимости ресурсов. Для оценки TCO введи тариф вычислений, объём запросов, время человека, стоимость данных и обслуживания.
8. Экспорт признаков по умолчанию выключен: TF-IDF матрицы и эмбеддинги могут кодировать чувствительные сведения. Joblib-модели следует загружать только из доверенных каталогов.

## Установка локально или в бесплатном облачном notebook

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e ".[test]"
```

Для необязательного E5:

```bash
pip install -e ".[embeddings]"
```

E5 потребует скачивания весов из Hugging Face и интернет-доступа. Это не должно вызывать платные расходы; при недоступности модели запускайте TF-IDF или устанавливайте веса заранее в доступной среде.

## Запуск в Google Colab

В архив включён ноутбук `notebooks/AMORPH_AutoTrainer_v0.1_Colab.ipynb`. Открой его в Google Colab и запускай ячейки по порядку. Ноутбук попросит загрузить ZIP проекта, установит пакет, выполнит тесты, запустит синтетический smoke test и проверит предсказание/извлечение признаков. Затем можно загрузить собственный CSV с колонками `text,label`.

Нужен интернет для установки Python-зависимостей. GPU не требуется для первого TF-IDF pipeline. Доступность бесплатной среды не гарантирована. Искусственные demo-данные предназначены только для проверки работы кода и не подтверждают качество на реальной задаче.

## Быстрый smoke test без скачивания моделей

```bash
amorph-autotrainer demo --output data/demo.csv --examples-per-class 30
amorph-autotrainer train data/demo.csv --text-column text --label-column label --out runs --strategies tfidf --max-runtime-seconds 7200 --max-candidates 1
```

Демо-данные искусственные. Высокая метрика на них не подтверждает качество на реальной задаче.

## Обучение на собственном CSV

CSV должен иметь столбцы `text,label` по умолчанию. Названия настраиваются.

```bash
amorph-autotrainer train my_data.csv --text-column text --label-column label --out runs --strategies tfidf --min-gate-examples-per-class 5
```

Если строки одного пользователя/источника/проекта не должны попадать в разные части, передай колонку с группой (она должна быть заполнена для всех используемых строк):

```bash
amorph-autotrainer train my_data.csv --text-column text --label-column label --group-column customer_id --out runs --strategies tfidf
```

Группировка по ID и хронологическое разделение решают разные задачи. Для оценки на будущих периодах не следует заменять отдельную временную политику случайным групповым split.

Опционально сравнить с E5 (требует установленные `sentence-transformers` и доступ к весам):

```bash
amorph-autotrainer train my_data.csv --text-column text --label-column label --out runs --strategies tfidf,e5 --max-candidates 2 --expected-queries 10000 --noninferiority-margin 0.01 --bootstrap-samples 500
```

Абсолютный порог качества задаётся явно, например:

```bash
--min-macro-f1 0.80
```

Это пример настройки, а не универсально подходящий порог.

## Получить предсказания и признаки

Путь модели — каталог `selected` из результата запуска:

```bash
amorph-autotrainer predict --model-dir runs/<run_id>/selected --input new_texts.csv --text-column text --output predictions.csv
amorph-autotrainer features --model-dir runs/<run_id>/selected --input new_texts.csv --text-column text --output features.npz
amorph-autotrainer runs --db runs/experiments.sqlite3 --limit 20
```

- Для TF-IDF `features.npz` — разреженная SciPy матрица.
- Для E5 `features.npz` — сжатый NPZ с dense массивом `features`.

## Политика выбора

Полная стоимость в этой версии оценивается по пользовательским ставкам, времени preprocessing/fit, измеренной задержке одного запроса, ожидаемому объёму запросов, стоимости данных, обслуживания и человеческого труда. Ставки по умолчанию равны нулю только для удобства smoke test; это не означает, что ресурсы и труд экономически бесплатны. Для RAG, LoRA/QLoRA и distillation пока нет исполняемых кандидатов, поэтому эти расходы не оцениваются автоматически.

- TF-IDF — обязательный дешёвый baseline.
- Кандидаты ранжируются **по validation**, а не по test.
- Если `--min-macro-f1` не задан, выбираются кандидаты в пределах `noninferiority-margin` от лучшего validation Macro-F1, а затем выбирается самый дешёвый по proxy `training_seconds + expected_queries * inference_seconds_per_query`.
- После выбора acceptance gate (отдельный split) сравнивает выбранного кандидата с TF-IDF baseline через paired bootstrap и проверяет predeclared `--min-macro-f1`, если он задан.
- Если альтернативный кандидат не проходит gate, сохраняется baseline. Если gate содержит меньше установленного минимума примеров класса, статус `acceptance_gate_underpowered`, CLI завершает работу с кодом 4 и не заявляет статистическую приёмку.
- Финальный test оценивается после выбора стратегии и не используется для fallback. После просмотра test нельзя подбирать настройки на тех же данных: следующая итерация требует нового независимого теста.
- Коды завершения: `0` — pipeline выполнен; `2` — заданный порог качества не пройден; `3` — превышен runtime budget; `4` — acceptance gate недостаточно информативен.

## Команды проверки

```bash
pytest
python -m compileall -q amorph_autotrainer
```

## Ближайшие расширения после проверки v0.1

1. Более сильная защита от почти-дубликатов и leakage.
2. Несколько seed / repeated holdout для оценки variability обучения.
3. Метрики retrieval: Recall@k, MRR, NDCG, если появились размеченные query-document пары.
4. Планировщик с Hyperband/Optuna только после наличия нескольких кандидатов, где это даёт измеримую экономию.
5. RAG, LoRA/QLoRA и distillation как реальные проверяемые стратегии — каждая со своей стоимостью и Quality Gate.
6. NCA-исследования только отдельным benchmark-ом против обычного retrieval/active learning.


## Воспроизводимый внешний benchmark: MASSIVE ru-RU

В проект добавлен `scripts/benchmark_massive_ru.py`. Скрипт скачивает официальные Parquet-файлы `AmazonScience/massive` по зафиксированному commit SHA, использует только официальный `train` для обучения/внутреннего выбора, удерживает официальный `test` до завершения выбора и считает внешние метрики после него. Для split внутри train используется `worker_id`; это полезный тест группового split, но не временная проверка и не универсальная оценка production-качества.

Условия запуска в среде с интернетом:

```bash
python -m pip install -e ".[test]"
python -m pip install pandas pyarrow huggingface_hub
python scripts/benchmark_massive_ru.py --output-dir benchmark_runs/massive_ru
```

Чтобы включить E5 и загрузку весов модели:

```bash
python -m pip install -e ".[embeddings]"
python scripts/benchmark_massive_ru.py --with-e5 --output-dir benchmark_runs/massive_ru_e5
```

Скрипт сохраняет `dataset_provenance.json` и `benchmark_result.json`, включая revision, локальные SHA-256, версии Python, внутренний результат AMORPH и внешние метрики на официальном тестовом наборе. Датасет опубликован под CC BY 4.0: сохраняйте атрибуцию и ссылку на карточку набора. MASSIVE — классификация интентов голосового ассистента, а не retrieval/RAG и не обучение генеративной LLM.

**Важно:** этот benchmark harness является воспроизводимым планом эксперимента, но сам по себе не означает, что benchmark уже был выполнен. Численные метрики на реальном MASSIVE нельзя заявлять, пока скрипт не завершился и результат не сохранён.

## Дополнения после P0 code review

- Для E5-классификации используется префикс `query: `, как рекомендует карточка `intfloat/multilingual-e5-small` для embeddings, применяемых как признаки.
- Исчерпание положительного runtime-лимита на ранних стадиях имеет отдельный тип исключения и возвращает CLI-код `3`; недопустимый `--max-runtime-seconds 0` по-прежнему считается ошибкой конфигурации и возвращает `1`.
- Confidence interval Quality Gate — paired class-stratified bootstrap по примерам. Если наблюдения внутри внешних групп коррелированы, это ещё не cluster bootstrap; интервал может быть слишком оптимистичным. Этот статистический пробел остаётся открытым.


## Экспериментальный профиль E5 для MASSIVE ru-RU

Для датасета MASSIVE ru-RU зафиксирован сильный кандидат C=0.1, class_weight=balanced; это результат одного воспроизводимого прогона, а не универсальная рекомендация. Production-default E5 в main остаётся C=1.0, class_weight=None. См. [отчёт эксперимента](docs/MASSIVE_RU_E5_CANDIDATE_2026-10-22.md) и [машиночитаемый профиль](configs/experimental/massive_ru_e5_c01_balanced.json). Официальный test — только отчётная оценка; профиль не включается автоматически.
