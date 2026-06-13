import pandas as pd
from typing import Dict, Optional

from src.data.loader import detect_column_mapping
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def clean_data(
    df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    if column_map is None:
        column_map = detect_column_mapping(df)

    discount_col = column_map.get("discount", "Discount")

    logger.info("Dropping rows with missing discount values")
    df = df.dropna(subset=[discount_col])

    logger.info("Cleaned data shape: %s", df.shape)
    return df


def get_column_map(
    df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    if column_map is None:
        column_map = detect_column_mapping(df)
    return column_map


def save_clean_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)
