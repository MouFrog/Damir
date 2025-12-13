from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(df, summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_new_quality_heuristics():
    # 1. Тест: has_constant_columns
    df1 = pd.DataFrame({
        "id": [1, 2, 3],
        "status": ["active", "active", "active"],  # константная колонка
        "value": [10, 20, 30]
    })
    summary1 = summarize_dataset(df1)
    missing_df1 = missing_table(df1)
    flags1 = compute_quality_flags(df1, summary1, missing_df1)
    assert flags1["has_constant_columns"] == True

    # 2. Тест: has_high_cardinality_categoricals
    # Порог в коде — 20 (или 50, зависит от вашего значения)
    # Создадим категориальную колонку с 25 уникальными значениями
    df2 = pd.DataFrame({
        "user_id": range(25),
        "category": [f"cat_{i}" for i in range(25)]  # 25 уникальных — выше порога
    })
    summary2 = summarize_dataset(df2)
    missing_df2 = missing_table(df2)
    flags2 = compute_quality_flags(df2, summary2, missing_df2)
    assert flags2["has_high_cardinality_categoricals"] == True

    # 3. Дополнительно: проверка, что флаги = False, когда не должно быть проблем
    df4 = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "city": ["Moscow", "SPb", "Novosibirsk", "Kazan"],  # 4 уникальных — низкая кардинальность
        "score": [1, 2, 3, 4]
    })
    summary4 = summarize_dataset(df4)
    missing_df4 = missing_table(df4)
    flags4 = compute_quality_flags(df4, summary4, missing_df4)
    assert flags4["has_constant_columns"] == False
    assert flags4["has_high_cardinality_categoricals"] == False