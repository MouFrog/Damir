# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## HW03

Команда report теперь поддерживает дополнительные параметры:

--max-hist-columns – сколько числовых колонок включать в гистограммы (по умолчанию: 6);
--top-k-categories – сколько top-значений выводить для категориальных признаков (по умолчанию: 5);
--title – заголовок отчёта в report.md;
--min-missing-share – порог доли пропусков (от 0 до 1), выше которого колонка считается проблемной.

## Тесты

```bash
uv run pytest -q
```

## Пример вызова

```bash
uv run eda-cli report data/example.csv \
  --out-dir my_report \
  --title "Анализ пользователей" \
  --top-k-categories 10 \
  --min-missing-share 0.2
```
