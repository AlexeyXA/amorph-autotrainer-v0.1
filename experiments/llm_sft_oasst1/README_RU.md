# Изолированный OASST1 → Qwen LoRA proof-of-concept

Это экспериментальная ветка, а не функция основного пакета AMORPH AutoTrainer v0.1. Основной код классификации и его настройки не меняются. Цель — проверить реальный SFT-процесс открытой LLM через LoRA, зафиксировать loss-кривые и применить отдельный acceptance gate.

## Зафиксированная конфигурация

- Модель Qwen/Qwen2.5-0.5B-Instruct, ревизия ec7ddfa904d4d447eedd0b7f126df16957734abb.
- Данные OpenAssistant/oasst1, ревизия fdf72ae0827c1cda404aff25b6603abec9e3399b.
- Отбираются русские пары prompter→assistant; ответ должен иметь положительный review_result и не быть помечен synthetic; берётся одна пара на message_tree_id.
- Upstream splits объединяются до создания собственного deterministic 70/10/10/10 разделения train/validation/gate/test по conversation tree. Ни одно дерево не должно пересекать две части.
- LoRA r=8, alpha=16, dropout=0.05; обучаются только q/k/v/o projection adapters, базовые веса заморожены.
- Потеря считается только по токенам ответа assistant; user prompt замаскирован.
- Метрика — невзвешенное среднее по деревьям из среднего answer-token NLL для каждой пары, nats/token; ниже лучше.
- Gate использует парный bootstrap по деревьям. ACCEPT разрешён, если верхняя 95%-я граница для NLL(adapter) − NLL(base) не выше заранее заданного допуска 0.02 nat/token. REJECT — если нижняя 95%-я граница выше допуска; иначе INCONCLUSIVE. Gate требует не менее 20 пригодных conversation trees.
- Сохраняются adapter, train/validation log, provenance, метрики и инвентарь SHA-256. Сырые диалоги в artifact не копируются.

## Первый фактический прогон

Отчёт: docs/LLM_SFT_OASST1_RUN_2026-10-23_RU.md.

- Workflow [#39960511458](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39960511458) завершился success; core CI и LoRA job прошли.
- Сохранён настоящий LoRA adapter, артефакт ID 12505021061; внутренние 12/12 файлов проверены по SHA-256 и размерам, digest ZIP совпадает с GitHub Actions.
- Gate получил только 19 пригодных деревьев вместо необходимых 20 и вернул INCONCLUSIVE. Это не принятие модели и не доказательство качества.
- Ограничение 256 токенов отбросило 58 из 128 train-кандидатов, а также 10/24 validation, 13/32 gate и 11/32 test примеров. Не использовать уже открытые final-test метрики для последующей настройки.

## Чего этот эксперимент не доказывает

Это маленькая техническая проверка: максимум 128 train examples, 24 validation, 32 gate, 32 test, 16 шагов и 256 токенов. Она не заменяет grid 100/500/2000 примеров × 1/2/3 эпох, повторные seed, независимые MERA/IFEval и проверки забывания на других задачах.

NLL на том же корпусе не доказывает качество ответов, безопасность, следование инструкциям или превосходство над ChatGPT, Grok либо Gemini. Этот код обучает adapter открытой модели; он не управляет закрытыми весами. Маленький gate может выдать INCONCLUSIVE — это не успех. Финальный test вычисляется только после решения gate; после просмотра его нельзя использовать для подбора параметров.

Фильтр текущего прототипа выбирает reviewed RU ответы и удаляет дублирующиеся message IDs, но это ещё не полноценный дедуп-аудит корпуса: не проверена полнота удаления synthetic-веток по каждому возможному родителю, и не проводится отдельный exact/near-duplicate аудит текста между разными conversation trees. Это ограничения текущего пилота, которые нужно закрыть до более крупного или исследовательского вывода; новые эксперименты после открытия test должны использовать заново предопределённый протокол и новый независимый holdout.

Карточки датасета и модели указывают Apache-2.0, однако лицензии, происхождение исходных данных, условия перераспространения и коммерческого использования следует перепроверить до соответствующего использования. Репозиторий публичный: доступ к workflow artifact следует считать доступным читателям репозитория. Не запускать эксперимент на частных данных без отдельного контроля доступа.

## Локальный запуск

Рекомендуется Python 3.12. Сначала установите CPU-only PyTorch, затем остальное:

    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
    python -m pip install -r experiments/llm_sft_oasst1/requirements.txt pytest==8.3.4
    python -m pytest -q tests/test_llm_sft_oasst1_data.py tests/test_llm_sft_oasst1_gate.py
    python -m experiments.llm_sft_oasst1.train --output-dir ./artifacts/oasst1-sft

Потребуется интернет для скачивания зафиксированного корпуса, tokenizer и весов. CPU используется намеренно, чтобы smoke test не зависел от платного GPU. Фактическую загрузку модели и обучение подтверждает конкретный успешный workflow, а не unit-тесты сами по себе.

## Ссылки

- OpenAssistant/oasst1: https://huggingface.co/datasets/OpenAssistant/oasst1
- Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Hugging Face TRL: https://huggingface.co/docs/trl/en/index
- MERA — будущая независимая русскоязычная оценка: https://mera.a-ai.ru/en/catalog
- IFEval — будущая оценка следования инструкциям: https://huggingface.co/datasets/google/IFEval
