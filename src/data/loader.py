import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    column_map: Dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "").replace("-", "")
        if "orderdate" in col_lower:
            column_map["date"] = col
        elif "productid" in col_lower:
            column_map["product_id"] = col
        elif col_lower in ("sales", "sale"):
            column_map["sales"] = col
        elif col_lower == "profit":
            column_map["profit"] = col
        elif col_lower == "discount":
            column_map["discount"] = col
        elif col_lower in ("quantity", "qty"):
            column_map["quantity"] = col

    logger.info("Detected column mapping: %s", column_map)
    return column_map


def load_raw_data(filepath: Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", filepath)
    if filepath.suffix in (".xls", ".xlsx"):
        df = pd.read_excel(filepath)
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    logger.info("Loaded data shape: %s", df.shape)
    return df


def load_cleaned_data(filepath: Path) -> pd.DataFrame:
    logger.info("Loading cleaned data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded cleaned data shape: %s", df.shape)
    return df


def load_feature_matrix(
    features_file: Path, target_file: Path
) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(features_file)
    y = pd.read_csv(target_file).squeeze("columns")
    logger.info("Loaded feature matrix X: %s, y: %s", X.shape, y.shape)
    return X, y
