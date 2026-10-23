# Отчёт о первом фактическом SFT/LoRA-прогоне AMORPH (2026-10-23)

## Статус

**Техническое обучение действительно выполнено и воспроизводимо проверено. Приёмка качества — INCONCLUSIVE.** Это первая техническая демонстрация внешнего training backend в отдельной экспериментальной ветке. Это не функция main, не доказательство общей способности чат-модели и не основание менять defaults AMORPH v0.1.

- PR: [#24 — Experimental Russian OASST1 SFT/LoRA proof of concept](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/pull/24).
- Успешный workflow: [run #39960511458](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39960511458).
- Commit, на котором выполнено обучение: b4be68d68d15c138dd4eea9a265ba613759a9303.
- Artifact ID: 12505021061, archive size 7,946,854 bytes, GitHub ZIP digest: sha256:f19d326aa4b8bd6312ad6ab93761364e39716e6bbebe0b90d5b044b689cac307.
- Архив содержит 13 файлов: 12 перечислены во внутреннем inventory и каждый прошёл повторную проверку SHA-256 и размера; тринадцатый файл — сам artifact_inventory.json, намеренно исключённый из собственного хеш-списка. Веса адаптера сохранены в adapter/adapter_model.safetensors.
- Ни исходные тексты диалогов, ни подготовленный dataset в артефакт не загружались.

## Закреплённые входы

- Dataset: OpenAssistant/oasst1, revision fdf72ae0827c1cda404aff25b6603abec9e3399b.
- Base model: Qwen/Qwen2.5-0.5B-Instruct, revision ec7ddfa904d4d447eedd0b7f126df16957734abb.
- Seed: 42.
- Загрузили 88,838 строк из upstream splits train + validation, до создания собственных разделов. После RU/review/direct-parent фильтра 4,153 пары прошло первичную проверку; выбран один prompt→assistant ответ на дерево, всего 760 conversation trees до split caps.
- Hash выбранных идентификаторов пар: 5e73d143195cc84f868011795ff5da7371fe4d6fc88e2c83864a2d4676ea09e3.
- Разделение: детерминированное 70/10/10/10 по message_tree_id; к каждому разделу применён заранее объявленный cap. Деревья были единицей split и bootstrap.
- Обучение: LoRA r=8, alpha=16, dropout=0.05, target modules q/k/v/o; базовые параметры заморожены; 16 шагов, CPU, max sequence length 256. Loss считается на assistant response tokens, prompt tokens masked.
- Сводная метрика: средний по conversation trees per-example answer-token NLL, nats/token; меньше — лучше. Это языковая модельная loss, не human preference/response-quality score.

## Что реально произошло

Поскольку часть выбранных примеров длиннее 256 токенов, несколько записей удалены до обучения/оценки:

| Раздел | Выбрано по деревьям | Отброшено из-за длины/пустого ответа | Фактически оценено/обучено |
|---|---:|---:|---:|
| Train | 128 | 58 | 70 |
| Validation | 24 | 10 | 14 |
| Acceptance gate | 32 | 13 | 19 |
| Final test | 32 | 11 | 21 |

Итого: train сохранил 70/128 примеров (54.7%); gate сохранил только 19 деревьев.

## Результаты

### Validation — диагностическая метрика, меньше лучше

| Модель | Средний answer-token NLL |
|---|---:|
| Base Qwen | 2.579923 |
| Qwen + LoRA | 2.449512 |
| Delta (adapter − base) | -0.130412 |

Значение validation улучшилось в этой маленькой выборке. Это полезный сигнал технической работоспособности, но оно не показывает, что ответы стали правильнее, полезнее или безопаснее.

### Acceptance gate — решение по заранее установленному правилу

- Парный bootstrap по conversation trees: запрошено 500 повторов.
- Минимум gate: 20 независимых деревьев.
- Фактически после length filter: **19**.
- Средняя разница NLL (adapter − base): -0.109322.
- CI lower / upper: отсутствуют, так как выборка ниже минимального порога.
- Verdict: **INCONCLUSIVE**, reason: underpowered_gate.

По правилам протокола этот результат нельзя интерпретировать как ACCEPT. Направление средней разницы благоприятно, но мощность/неопределённость не позволяют сделать вывод о non-inferiority.

### Final test — report-only, уже раскрыт

| Модель | Средний answer-token NLL |
|---|---:|
| Base Qwen | 2.384893 |
| Qwen + LoRA | 2.309505 |
| Delta (adapter − base) | -0.075388 |
| Число пригодных деревьев | 21 |

Эти значения публикуются только для полноты отчёта. Теперь final-test метрики наблюдались; **нельзя использовать этот же split для выбора параметров или доказательства последующего улучшения**. Следующий подтверждающий запуск требует нового заранее объявленного holdout либо другой независимой ревизии/выборки.

## Проверка артефактов

ZIP был загружен через GitHub Actions, затем скачан и независимо проверен:

- SHA-256 ZIP равен GitHub Actions digest: f19d326aa4b8bd6312ad6ab93761364e39716e6bbebe0b90d5b044b689cac307.
- Все 12 записей в artifact_inventory.json проверены: путь существует, размер совпадает, SHA-256 совпадает.
- Обнаружен adapter/adapter_model.safetensors (4,350,392 bytes).
- Присутствуют training_log.json, dataset_provenance.json, benchmark_result.json, artifact_inventory.json.
- Сырые диалоги и подготовленный dataset отсутствуют в артефакте.

## Ограничения и следующие исправления

1. **Тестовая выборка мала.** 16 шагов на 70 пригодных train примерах — это технический smoke test, не полноценный эксперимент над generalization.
2. **Length filtering слишком агрессивен.** В следующем протоколе нужно заранее увеличить max length/выбрать правило сокращения с сохранением ответа и получить как минимум 20 пригодных независимых gate trees; желательно заметно больше, с отдельным power analysis.
3. **Дедупликация не завершена.** Прототип удаляет конфликтующие дубли message IDs и берёт одну пару на дерево, но не делает отдельный exact/near-duplicate audit текста между различными деревьями; synthetic-флаг родительского prompt также не проверяется полностью. Нужны отдельные тесты/правки до более крупного эксперимента.
4. **Внешняя оценка отсутствует.** MERA/IFEval, задачи вне train и regression/forgetting suite не запускались.
5. **Переобучение не проверено полноценно.** Train/eval loss записывался каждые четыре шага, но это не сетка 100/500/2000 примеров × 1/2/3 эпох и не повторение seed.
6. **Закрытые сервисы не обучались.** Веса ChatGPT, Grok и Gemini недоступны этому pipeline; обучался только LoRA-adapter открытой модели Qwen.
7. **Scope изолирован.** В core AMORPH v0.1 не добавляли SFT/LoRA и не меняли селектор или defaults.

**Следующий правильный эксперимент:** улучшить data audit и длину sequence на отдельной новой версии протокола, зафиксировать split/конфигурацию до запуска, оценить base vs adapter на достаточном acceptance gate и выполнить независимый instruction-following/regression benchmark. Не подбирать гиперпараметры по уже открытому final test из run #39960511458.
