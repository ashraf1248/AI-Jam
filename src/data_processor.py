from __future__ import annotations

import io

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.schemas import DatasetCorrelation, DatasetSummary, RegressionInsight


def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _strongest_correlations(frame: pd.DataFrame) -> list[DatasetCorrelation]:
    numeric = frame.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr(numeric_only=True).fillna(0.0)
    pairs: list[DatasetCorrelation] = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            pairs.append(
                DatasetCorrelation(
                    column_a=left,
                    column_b=right,
                    correlation=float(corr.loc[left, right]),
                )
            )
    return sorted(pairs, key=lambda item: abs(item.correlation), reverse=True)[:5]


def _outlier_counts(frame: pd.DataFrame) -> dict[str, int]:
    outliers: dict[str, int] = {}
    numeric = frame.select_dtypes(include=["number"])
    for column in numeric.columns:
        series = numeric[column].dropna()
        if len(series) < 4:
            outliers[column] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers[column] = 0
            continue
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        outliers[column] = int(mask.sum())
    return outliers


def _regression_insight(frame: pd.DataFrame) -> RegressionInsight | None:
    numeric = frame.select_dtypes(include=["number"]).dropna()
    if numeric.shape[1] < 2 or len(numeric) < 5:
        return None
    target_column = numeric.columns[-1]
    feature_columns = [column for column in numeric.columns if column != target_column]
    if not feature_columns:
        return None
    x = numeric[feature_columns]
    y = numeric[target_column]
    model = LinearRegression()
    model.fit(x, y)
    importance = {
        column: float(abs(weight))
        for column, weight in zip(feature_columns, model.coef_, strict=True)
    }
    return RegressionInsight(
        target_column=target_column,
        feature_importance=importance,
        r2_score=float(model.score(x, y)),
    )


def _plain_english_summary(
    filename: str,
    frame: pd.DataFrame,
    strongest_correlations: list[DatasetCorrelation],
    outlier_counts: dict[str, int],
    regression_insight: RegressionInsight | None,
) -> str:
    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    categorical_columns = list(frame.select_dtypes(exclude=["number"]).columns)
    parts = [
        f"{filename} contains {frame.shape[0]} rows and {frame.shape[1]} columns.",
        f"Numeric columns: {', '.join(numeric_columns) if numeric_columns else 'none detected'}.",
        f"Categorical columns: {', '.join(categorical_columns) if categorical_columns else 'none detected'}.",
    ]
    if strongest_correlations:
        top = strongest_correlations[0]
        parts.append(
            f"The strongest numeric relationship is between {top.column_a} and {top.column_b} "
            f"(correlation {top.correlation:.2f})."
        )
    notable_outliers = [f"{column} ({count})" for column, count in outlier_counts.items() if count]
    if notable_outliers:
        parts.append("Possible outliers appear in " + ", ".join(notable_outliers) + ".")
    if regression_insight and regression_insight.feature_importance:
        main_feature = max(
            regression_insight.feature_importance,
            key=regression_insight.feature_importance.get,
        )
        parts.append(
            f"A simple regression suggests {main_feature} is most associated with "
            f"{regression_insight.target_column}."
        )
    return " ".join(parts)


def analyze_dataframe(frame: pd.DataFrame, filename: str) -> DatasetSummary:
    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    categorical_columns = list(frame.select_dtypes(exclude=["number"]).columns)
    descriptive_stats: dict[str, dict[str, float | int | None]] = {}
    if numeric_columns:
        stats = frame[numeric_columns].describe().replace({np.nan: None})
        descriptive_stats = {
            column: {stat: (None if pd.isna(value) else float(value)) for stat, value in values.items()}
            for column, values in stats.to_dict().items()
        }

    strongest = _strongest_correlations(frame)
    outliers = _outlier_counts(frame)
    regression = _regression_insight(frame)
    summary = _plain_english_summary(filename, frame, strongest, outliers, regression)

    return DatasetSummary(
        filename=filename,
        shape=(int(frame.shape[0]), int(frame.shape[1])),
        columns=[str(column) for column in frame.columns.tolist()],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_values={str(key): int(value) for key, value in frame.isna().sum().items()},
        descriptive_statistics=descriptive_stats,
        strongest_correlations=strongest,
        outlier_counts=outliers,
        regression_insight=regression,
        plain_english_summary=summary,
    )


def analyze_csv_bytes(file_bytes: bytes, filename: str) -> DatasetSummary:
    frame = load_csv_bytes(file_bytes)
    return analyze_dataframe(frame, filename)
