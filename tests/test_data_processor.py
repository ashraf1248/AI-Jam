import pandas as pd

from src.data_processor import analyze_dataframe


def test_analyze_dataframe_handles_small_numeric_and_categorical_data() -> None:
    frame = pd.DataFrame(
        {
            "temperature": [20, 21, 22, 23, 24, 25],
            "yield": [10, 10.5, 11, 11.3, 12.1, 12.6],
            "condition": ["A", "A", "B", "B", "B", "A"],
        }
    )
    summary = analyze_dataframe(frame, "sample.csv")
    assert summary.filename == "sample.csv"
    assert summary.shape == (6, 3)
    assert "temperature" in summary.numeric_columns
    assert "condition" in summary.categorical_columns
    assert summary.plain_english_summary
