import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.data.loader import detect_column_mapping
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def build_features(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if column_map is None:
        column_map = detect_column_mapping(df)

    if "date" not in column_map:
        raise KeyError("Date column not found in dataset")

    date_col = column_map["date"]
    product_col = column_map.get("product_id")
    sales_col = column_map.get("sales", "Sales")
    profit_col = column_map.get("profit", "Profit")
    qty_col = column_map.get("quantity", "Quantity")
    discount_col = column_map.get("discount", "Discount")

    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_col])
    df["_product"] = df[product_col] if product_col else "DEFAULT"
    df["_sales"] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
    df["_profit"] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0)
    df["_qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df["_discount"] = pd.to_numeric(df[discount_col], errors="coerce").fillna(0)

    df["profit_margin"] = df["_profit"] / df["_sales"].replace(0, np.nan)
    df["profit_margin"] = df["profit_margin"].fillna(0).clip(0, 1)

    df["day_of_week"] = df["_date"].dt.dayofweek
    df["month"] = df["_date"].dt.month
    df["quarter"] = df["_date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.sort_values(["_product", "_date"])
    df["days_since_last_sale"] = (
        df.groupby("_product")["_date"].diff().dt.days
    )
    df["days_since_last_sale"] = df["days_since_last_sale"].fillna(365)

    df["sales_velocity_7d"] = df.groupby("_product")["_qty"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df["current_stock"] = df["_qty"]
    df["sales_velocity_30d"] = df.groupby("_product")["_qty"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    result_cols = feature_columns or [
        "sales_velocity_7d",
        "sales_velocity_30d",
        "days_since_last_sale",
        "profit_margin",
        "day_of_week",
        "month",
        "quarter",
        "is_weekend",
        "current_stock",
    ]

    result = df[result_cols].fillna(0)
    logger.info(
        "Built feature matrix with shape %s using columns: %s",
        result.shape,
        result_cols,
    )
    return result


def build_features_from_input(
    records: List[Dict], feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    cols = feature_columns or [
        "sales_velocity_7d",
        "sales_velocity_30d",
        "days_since_last_sale",
        "profit_margin",
        "day_of_week",
        "month",
        "quarter",
        "is_weekend",
        "current_stock",
    ]
    rows = []
    for rec in records:
        row = [rec.get(k, 0) for k in cols]
        rows.append(row)

    result = pd.DataFrame(rows, columns=cols)
    logger.info("Built prediction feature matrix with shape %s", result.shape)
    return result
