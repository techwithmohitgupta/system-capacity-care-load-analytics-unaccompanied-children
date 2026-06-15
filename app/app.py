# ============================================================
# System Capacity & Care Load Analytics for Unaccompanied Children
# Production-Ready Streamlit + Plotly Healthcare Analytics Dashboard
# Phase 1: Complete app.py Replacement
# ============================================================

from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageChops
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ============================================================
# 1. Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="System Capacity & Care Load Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. Project Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

ASSETS_DIR = APP_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

CSS_PATH = ASSETS_DIR / "hhs_dashboard.css"
HHS_LOGO_PATH = IMAGES_DIR / "hhs_logo.png"
UNIFIED_MENTOR_LOGO_PATH = IMAGES_DIR / "unified_mentor.png"

PRIMARY_PROCESSED_DIR = PROJECT_ROOT / "notebooks" / "data" / "processed"
SECONDARY_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_FILE_CANDIDATES = {
    "main": [
        PRIMARY_PROCESSED_DIR / "final_pressure_stress_classification_dataset.csv",
        SECONDARY_PROCESSED_DIR / "final_pressure_stress_classification_dataset.csv",
    ],
    "kpi_cards": [
        PRIMARY_PROCESSED_DIR / "06_kpi_dashboard_cards.csv",
        SECONDARY_PROCESSED_DIR / "06_kpi_dashboard_cards.csv",
    ],
    "final_kpi_summary": [
        PRIMARY_PROCESSED_DIR / "06_final_kpi_summary.csv",
        SECONDARY_PROCESSED_DIR / "06_final_kpi_summary.csv",
    ],
    "trend_direction": [
        PRIMARY_PROCESSED_DIR / "06_trend_direction_dataset.csv",
        SECONDARY_PROCESSED_DIR / "06_trend_direction_dataset.csv",
    ],
}


# ============================================================
# 3. Design System and Plotly Configuration
# ============================================================

COLORS = {
    "background": "#f5f9fc",
    "card": "#ffffff",
    "navy": "#102a43",
    "blue": "#2563eb",
    "blue_soft": "#60a5fa",
    "teal": "#0f766e",
    "teal_soft": "#2dd4bf",
    "green": "#16a34a",
    "amber": "#f59e0b",
    "orange": "#f97316",
    "red": "#dc2626",
    "slate": "#64748b",
    "slate_light": "#cbd5e1",
    "gray": "#94a3b8",
    "purple": "#6366f1",
}

PRESSURE_COLORS = {
    "low": COLORS["green"],
    "stable": COLORS["green"],
    "normal": COLORS["teal"],
    "moderate": COLORS["amber"],
    "medium": COLORS["amber"],
    "elevated": COLORS["orange"],
    "high": COLORS["orange"],
    "severe": COLORS["red"],
    "critical": COLORS["red"],
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

STANDARD_HEIGHT = 430
COMPACT_HEIGHT = 350


# ============================================================
# 4. CSS and Data Loading
# ============================================================

def load_css(css_path: Path) -> None:
    """Load external CSS without exposing developer/debug sections."""
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def first_existing_path(candidates: list[Path]) -> Path | None:
    """Return the first existing file path from a candidate list."""
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_csv_safely(path_text: str) -> pd.DataFrame:
    """Load a CSV safely with automatic date parsing for known date-like columns."""
    path = Path(path_text)

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to load `{path.name}`. Error: {exc}")
        return pd.DataFrame()

    for column in df.columns:
        normalized = normalize_text(column)
        if normalized in {
            "date",
            "reporting_date",
            "period",
            "trend_period",
            "period_start_date",
            "period_end_date",
        }:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def load_optional_dataset(dataset_key: str) -> pd.DataFrame:
    """Load optional supporting processed datasets."""
    path = first_existing_path(DATA_FILE_CANDIDATES.get(dataset_key, []))
    if path is None:
        return pd.DataFrame()
    return load_csv_safely(str(path))


# ============================================================
# 5. Column Mapping and Data Preparation
# ============================================================

def normalize_text(value: Any) -> str:
    """Normalize text for resilient column matching."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("%", " pct")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
    )


def normalized_key(value: Any) -> str:
    """Create a strict normalized key for exact matching."""
    return (
        normalize_text(value)
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column using exact normalized matching."""
    if df.empty:
        return None

    column_map = {normalized_key(column): column for column in df.columns}

    for candidate in candidates:
        key = normalized_key(candidate)
        if key in column_map:
            return column_map[key]

    return None


def find_column_contains(df: pd.DataFrame, required_terms: list[str]) -> str | None:
    """Find first column containing all required terms."""
    if df.empty:
        return None

    terms = [normalize_text(term) for term in required_terms]

    for column in df.columns:
        normalized = normalize_text(column)
        if all(term in normalized for term in terms):
            return column

    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


def assign_canonical_column(
    df: pd.DataFrame,
    canonical_name: str,
    source_candidates: list[str],
) -> pd.DataFrame:
    """Assign a canonical dashboard column when a source column exists."""
    source_column = find_column(df, source_candidates)

    if source_column and canonical_name not in df.columns:
        df[canonical_name] = df[source_column]

    return df


def prepare_dashboard_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the final analytical dashboard dataset."""
    if raw_df.empty:
        return raw_df.copy()

    df = raw_df.copy()

    canonical_mappings = {
        "date": [
            "Date",
            "date",
            "reporting_date",
        ],
        "children_apprehended_cbp": [
            "Children apprehended and placed in CBP custody",
            "children_apprehended_and_placed_in_cbp_custody",
            "children_apprehended_cbp",
            "daily_intake_volume",
        ],
        "cbp_custody": [
            "Children in CBP custody",
            "children_in_cbp_custody",
            "cbp_custody",
            "active_cbp_care_load",
        ],
        "transfers_to_hhs": [
            "Children transferred out of CBP custody",
            "children_transferred_out_of_cbp_custody",
            "transfers_to_hhs",
            "flow_into_hhs_system",
        ],
        "hhs_care": [
            "Children in HHS Care",
            "children_in_hhs_care",
            "hhs_care",
            "active_hhs_care_load",
        ],
        "hhs_discharged": [
            "Children discharged from HHS Care",
            "children_discharged_from_hhs_care",
            "hhs_discharged",
            "discharges_from_hhs",
            "successful_sponsor_placements",
        ],
        "total_system_load": [
            "total_system_load",
            "Total System Load",
            "total_children_under_care",
        ],
        "net_daily_intake": [
            "net_daily_intake",
            "Net Daily Intake",
            "net_intake_pressure",
        ],
        "care_load_growth_rate_pct": [
            "care_load_growth_rate_pct",
            "Care Load Growth Rate %",
            "care_load_growth_rate",
        ],
        "positive_net_intake_flag": [
            "positive_net_intake_flag",
            "Positive Net Intake Flag",
        ],
        "backlog_indicator": [
            "backlog_indicator",
            "Backlog Indicator",
        ],
        "backlog_severity_score": [
            "backlog_severity_score",
            "Backlog Severity Score",
        ],
        "rolling_7d_avg_system_load": [
            "rolling_7d_avg_system_load",
            "rolling_7_day_avg_system_load",
            "rolling_7d_system_load",
        ],
        "rolling_14d_avg_system_load": [
            "rolling_14d_avg_system_load",
            "rolling_14_day_avg_system_load",
            "rolling_14d_system_load",
        ],
        "rolling_7d_avg_net_intake": [
            "rolling_7d_avg_net_intake",
            "rolling_7_day_avg_net_intake",
        ],
        "rolling_14d_avg_net_intake": [
            "rolling_14d_avg_net_intake",
            "rolling_14_day_avg_net_intake",
        ],
        "final_pressure_stress_score": [
            "final_pressure_stress_score",
            "pressure_stress_score",
            "pressure_score",
        ],
        "final_pressure_stress_level": [
            "final_pressure_stress_level",
            "pressure_stress_level",
            "pressure_level",
        ],
        "primary_pressure_driver": [
            "primary_pressure_driver",
            "pressure_driver",
        ],
        "operational_response_category": [
            "operational_response_category",
            "response_category",
        ],
        "prolonged_pressure_window_flag": [
            "prolonged_pressure_window_flag",
            "prolonged_strain_window_flag",
        ],
        "high_pressure_day_flag": [
            "high_pressure_day_flag",
        ],
        "severe_pressure_day_flag": [
            "severe_pressure_day_flag",
        ],
    }

    for canonical_name, candidates in canonical_mappings.items():
        df = assign_canonical_column(df, canonical_name, candidates)

    if "date" not in df.columns:
        date_column = find_column_contains(df, ["date"])
        if date_column:
            df["date"] = df[date_column]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_candidates = [
        "children_apprehended_cbp",
        "cbp_custody",
        "transfers_to_hhs",
        "hhs_care",
        "hhs_discharged",
        "total_system_load",
        "net_daily_intake",
        "care_load_growth_rate_pct",
        "positive_net_intake_flag",
        "backlog_severity_score",
        "rolling_7d_avg_system_load",
        "rolling_14d_avg_system_load",
        "rolling_7d_avg_net_intake",
        "rolling_14d_avg_net_intake",
        "final_pressure_stress_score",
        "prolonged_pressure_window_flag",
        "high_pressure_day_flag",
        "severe_pressure_day_flag",
    ]

    for column in numeric_candidates:
        if column in df.columns:
            df[column] = coerce_numeric(df[column])

    if "total_system_load" not in df.columns and {"cbp_custody", "hhs_care"}.issubset(df.columns):
        df["total_system_load"] = df["cbp_custody"] + df["hhs_care"]

    if "net_daily_intake" not in df.columns and {"transfers_to_hhs", "hhs_discharged"}.issubset(df.columns):
        df["net_daily_intake"] = df["transfers_to_hhs"] - df["hhs_discharged"]

    if "care_load_growth_rate_pct" not in df.columns and "total_system_load" in df.columns:
        df["care_load_growth_rate_pct"] = df["total_system_load"].pct_change() * 100

    if "positive_net_intake_flag" not in df.columns and "net_daily_intake" in df.columns:
        df["positive_net_intake_flag"] = (df["net_daily_intake"] > 0).astype(int)

    if "backlog_severity_score" not in df.columns and "positive_net_intake_flag" in df.columns:
        df["backlog_severity_score"] = (
            df["positive_net_intake_flag"]
            .groupby((df["positive_net_intake_flag"] != df["positive_net_intake_flag"].shift()).cumsum())
            .cumsum()
        )

    if "rolling_7d_avg_system_load" not in df.columns and "total_system_load" in df.columns:
        df["rolling_7d_avg_system_load"] = df["total_system_load"].rolling(7, min_periods=1).mean()

    if "rolling_14d_avg_system_load" not in df.columns and "total_system_load" in df.columns:
        df["rolling_14d_avg_system_load"] = df["total_system_load"].rolling(14, min_periods=1).mean()

    if "rolling_7d_avg_net_intake" not in df.columns and "net_daily_intake" in df.columns:
        df["rolling_7d_avg_net_intake"] = df["net_daily_intake"].rolling(7, min_periods=1).mean()

    if "rolling_14d_avg_net_intake" not in df.columns and "net_daily_intake" in df.columns:
        df["rolling_14d_avg_net_intake"] = df["net_daily_intake"].rolling(14, min_periods=1).mean()

    return df


def required_columns_available(df: pd.DataFrame) -> bool:
    """Check minimum columns needed for the guideline-based dashboard."""
    required = {"date", "cbp_custody", "hhs_care", "transfers_to_hhs", "hhs_discharged"}
    return required.issubset(set(df.columns))


# ============================================================
# 6. Formatting and Analytics Helpers
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_compact_number(value: Any, decimals: int = 0) -> str:
    """Format a value for KPI cards."""
    number = safe_float(value)

    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"{number:,.{decimals}f}" if decimals else f"{number:,.0f}"
    return f"{number:.{decimals}f}" if decimals else f"{number:,.0f}"


def format_pct(value: Any, decimals: int = 1) -> str:
    """Format percent values."""
    return f"{safe_float(value):.{decimals}f}%"


def format_ratio(value: Any, decimals: int = 2) -> str:
    """Format ratio values."""
    return f"{safe_float(value):.{decimals}f}x"


def latest_value(df: pd.DataFrame, column: str) -> float:
    """Return latest value from a sorted dataframe column."""
    if df.empty or column not in df.columns:
        return 0.0
    return safe_float(df.sort_values("date")[column].dropna().tail(1).iloc[0]) if not df[column].dropna().empty else 0.0


def previous_value(df: pd.DataFrame, column: str) -> float:
    """Return previous value from a sorted dataframe column."""
    if df.empty or column not in df.columns:
        return 0.0

    values = df.sort_values("date")[column].dropna()
    if len(values) < 2:
        return latest_value(df, column)

    return safe_float(values.iloc[-2])


def calculate_kpis(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate five guideline-required KPIs dynamically from the filtered dataset."""
    if df.empty:
        return {
            "total_children_under_care": {
                "label": "Total Children Under Care",
                "value": 0.0,
                "display": "0",
                "delta": "0",
                "status": "No Data",
                "meaning": "System-wide responsibility across CBP custody and HHS care.",
            },
            "net_intake_pressure": {
                "label": "Net Intake Pressure",
                "value": 0.0,
                "display": "0 / day",
                "delta": "0",
                "status": "No Data",
                "meaning": "Balance between HHS transfers and HHS discharges.",
            },
            "care_load_volatility_index": {
                "label": "Care Load Volatility Index",
                "value": 0.0,
                "display": "0.0%",
                "delta": "0",
                "status": "No Data",
                "meaning": "Stability of total care load over the selected period.",
            },
            "backlog_accumulation_rate": {
                "label": "Backlog Accumulation Rate",
                "value": 0.0,
                "display": "0.0%",
                "delta": "0",
                "status": "No Data",
                "meaning": "Share of selected days with sustained positive care pressure.",
            },
            "discharge_offset_ratio": {
                "label": "Discharge Offset Ratio",
                "value": 0.0,
                "display": "0.00x",
                "delta": "0",
                "status": "No Data",
                "meaning": "Ability of HHS discharges to offset incoming transfers.",
            },
        }

    sorted_df = df.sort_values("date").copy()

    current_total_load = latest_value(sorted_df, "total_system_load")
    prior_total_load = previous_value(sorted_df, "total_system_load")
    total_load_delta = current_total_load - prior_total_load

    avg_net_intake = (
        safe_float(sorted_df["net_daily_intake"].mean())
        if "net_daily_intake" in sorted_df.columns
        else 0.0
    )

    latest_net_intake = latest_value(sorted_df, "net_daily_intake")
    prior_net_intake = previous_value(sorted_df, "net_daily_intake")
    net_intake_delta = latest_net_intake - prior_net_intake

    if "care_load_growth_rate_pct" in sorted_df.columns:
        volatility_index = safe_float(sorted_df["care_load_growth_rate_pct"].dropna().std())
    elif "total_system_load" in sorted_df.columns:
        volatility_index = safe_float(sorted_df["total_system_load"].pct_change().dropna().std() * 100)
    else:
        volatility_index = 0.0

    if "positive_net_intake_flag" in sorted_df.columns:
        backlog_rate = safe_float(sorted_df["positive_net_intake_flag"].fillna(0).mean() * 100)
    elif "net_daily_intake" in sorted_df.columns:
        backlog_rate = safe_float((sorted_df["net_daily_intake"] > 0).mean() * 100)
    else:
        backlog_rate = 0.0

    total_transfers = (
        safe_float(sorted_df["transfers_to_hhs"].sum())
        if "transfers_to_hhs" in sorted_df.columns
        else 0.0
    )
    total_discharges = (
        safe_float(sorted_df["hhs_discharged"].sum())
        if "hhs_discharged" in sorted_df.columns
        else 0.0
    )
    discharge_offset_ratio = total_discharges / total_transfers if total_transfers > 0 else 0.0

    if avg_net_intake > 0:
        net_status = "Pressure Building"
    elif avg_net_intake < 0:
        net_status = "Relief Improving"
    else:
        net_status = "Balanced"

    if volatility_index <= 2:
        volatility_status = "Stable"
    elif volatility_index <= 5:
        volatility_status = "Watch"
    else:
        volatility_status = "Volatile"

    if backlog_rate <= 40:
        backlog_status = "Controlled"
    elif backlog_rate <= 60:
        backlog_status = "Watch"
    else:
        backlog_status = "Accumulating"

    if discharge_offset_ratio >= 1:
        offset_status = "Relief Ahead"
    elif discharge_offset_ratio >= 0.85:
        offset_status = "Near Balance"
    else:
        offset_status = "Pressure Gap"

    return {
        "total_children_under_care": {
            "label": "Total Children Under Care",
            "value": current_total_load,
            "display": format_compact_number(current_total_load),
            "delta": f"{total_load_delta:+,.0f} vs prior day",
            "status": "Current Load",
            "meaning": "System-wide responsibility across CBP custody and HHS care.",
        },
        "net_intake_pressure": {
            "label": "Net Intake Pressure",
            "value": avg_net_intake,
            "display": f"{format_compact_number(avg_net_intake, 1)} / day",
            "delta": f"{net_intake_delta:+,.0f} latest change",
            "status": net_status,
            "meaning": "Average difference between transfers into HHS and discharges from HHS.",
        },
        "care_load_volatility_index": {
            "label": "Care Load Volatility Index",
            "value": volatility_index,
            "display": format_pct(volatility_index, 1),
            "delta": volatility_status,
            "status": volatility_status,
            "meaning": "Standard deviation of care load growth rate in the selected period.",
        },
        "backlog_accumulation_rate": {
            "label": "Backlog Accumulation Rate",
            "value": backlog_rate,
            "display": format_pct(backlog_rate, 1),
            "delta": backlog_status,
            "status": backlog_status,
            "meaning": "Percentage of selected days where net intake pressure was positive.",
        },
        "discharge_offset_ratio": {
            "label": "Discharge Offset Ratio",
            "value": discharge_offset_ratio,
            "display": format_ratio(discharge_offset_ratio, 2),
            "delta": offset_status,
            "status": offset_status,
            "meaning": "Total HHS discharges divided by total transfers into HHS.",
        },
    }


def build_kpi_interpretation_table(kpis: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create an analytical KPI interpretation table."""
    rows = []

    for item in kpis.values():
        rows.append(
            {
                "KPI": item["label"],
                "Current Value": item["display"],
                "Status": item["status"],
                "Operational Meaning": item["meaning"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 7. Filtering and Aggregation
# ============================================================

def apply_date_filter(df: pd.DataFrame, selected_range: tuple[Any, Any] | list[Any] | None) -> pd.DataFrame:
    """Apply date range filtering."""
    if df.empty or "date" not in df.columns or not selected_range or len(selected_range) != 2:
        return df.copy()

    start_date, end_date = selected_range

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()


def mode_or_last(series: pd.Series) -> Any:
    """Return mode if available, otherwise latest non-null value."""
    clean_series = series.dropna()
    if clean_series.empty:
        return np.nan

    mode_values = clean_series.mode()
    if not mode_values.empty:
        return mode_values.iloc[0]

    return clean_series.iloc[-1]


def aggregate_by_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate data according to selected time granularity."""
    if df.empty or "date" not in df.columns:
        return df.copy()

    if granularity == "Daily":
        return df.sort_values("date").copy()

    df_indexed = df.sort_values("date").set_index("date")

    rule = "W" if granularity == "Weekly" else "ME"

    aggregation_map: dict[str, Any] = {}

    flow_columns = [
        "children_apprehended_cbp",
        "transfers_to_hhs",
        "hhs_discharged",
        "net_daily_intake",
    ]

    load_columns = [
        "cbp_custody",
        "hhs_care",
        "total_system_load",
        "rolling_7d_avg_system_load",
        "rolling_14d_avg_system_load",
        "rolling_7d_avg_net_intake",
        "rolling_14d_avg_net_intake",
        "care_load_growth_rate_pct",
        "final_pressure_stress_score",
        "backlog_severity_score",
        "positive_net_intake_flag",
        "prolonged_pressure_window_flag",
        "high_pressure_day_flag",
        "severe_pressure_day_flag",
    ]

    categorical_columns = [
        "final_pressure_stress_level",
        "primary_pressure_driver",
        "operational_response_category",
        "backlog_indicator",
    ]

    for column in flow_columns:
        if column in df_indexed.columns:
            aggregation_map[column] = "sum"

    for column in load_columns:
        if column in df_indexed.columns:
            aggregation_map[column] = "mean"

    for column in categorical_columns:
        if column in df_indexed.columns:
            aggregation_map[column] = mode_or_last

    aggregated = df_indexed.resample(rule).agg(aggregation_map).reset_index()

    if {"cbp_custody", "hhs_care"}.issubset(aggregated.columns):
        aggregated["total_system_load"] = aggregated["cbp_custody"] + aggregated["hhs_care"]

    if {"transfers_to_hhs", "hhs_discharged"}.issubset(aggregated.columns):
        aggregated["net_daily_intake"] = aggregated["transfers_to_hhs"] - aggregated["hhs_discharged"]

    if "total_system_load" in aggregated.columns:
        aggregated["care_load_growth_rate_pct"] = aggregated["total_system_load"].pct_change() * 100
        aggregated["rolling_7d_avg_system_load"] = aggregated["total_system_load"].rolling(7, min_periods=1).mean()
        aggregated["rolling_14d_avg_system_load"] = aggregated["total_system_load"].rolling(14, min_periods=1).mean()

    if "net_daily_intake" in aggregated.columns:
        aggregated["positive_net_intake_flag"] = (aggregated["net_daily_intake"] > 0).astype(int)
        aggregated["rolling_7d_avg_net_intake"] = aggregated["net_daily_intake"].rolling(7, min_periods=1).mean()
        aggregated["rolling_14d_avg_net_intake"] = aggregated["net_daily_intake"].rolling(14, min_periods=1).mean()

    return aggregated.sort_values("date").reset_index(drop=True)


# ============================================================
# Module 1 — System Load Overview Command Center Intelligence
# Patch 1: Intelligence, Status, Insight, and Recommendation Logic
# ============================================================

def get_granularity_context(selected_granularity: str) -> dict[str, str]:
    """Return period-aware wording for daily, weekly, and monthly views."""
    granularity = str(selected_granularity or "Daily").strip().lower()

    if granularity == "weekly":
        return {
            "period_name": "week",
            "period_label": "weekly",
            "previous_label": "vs prior week",
            "movement_label": "weekly movement",
            "average_label": "average per week",
        }

    if granularity == "monthly":
        return {
            "period_name": "month",
            "period_label": "monthly",
            "previous_label": "vs prior month",
            "movement_label": "monthly movement",
            "average_label": "average per month",
        }

    return {
        "period_name": "day",
        "period_label": "daily",
        "previous_label": "vs prior day",
        "movement_label": "daily movement",
        "average_label": "average per day",
    }


def classify_module1_status(
    current_load: float,
    average_load: float,
    peak_load: float,
    load_delta_pct: float,
    net_intake_pressure: float,
    volatility_index: float,
    backlog_pressure_share: float,
    rolling_direction: str,
) -> dict[str, Any]:
    """Classify Module 1 operational status using combined capacity, trend, pressure, and risk signals."""
    if peak_load <= 0 or average_load <= 0:
        return {
            "status": "No Data",
            "severity": "neutral",
            "status_rank": 0,
            "summary": "Insufficient data is available for the selected period.",
            "leadership_focus": "Validate date range and dataset availability before interpreting system load.",
        }

    load_position_pct = (current_load / peak_load) * 100 if peak_load else 0.0
    above_average = current_load >= average_load
    near_peak = load_position_pct >= 90
    elevated_range = load_position_pct >= 75
    rising_load = load_delta_pct > 5
    falling_load = load_delta_pct < -5
    pressure_building = net_intake_pressure > 0
    relief_improving = net_intake_pressure < 0
    volatile = volatility_index > 5
    watch_volatility = 2 < volatility_index <= 5
    backlog_accumulating = backlog_pressure_share > 60
    backlog_watch = 40 < backlog_pressure_share <= 60
    rolling_rising = rolling_direction == "Rising"
    rolling_easing = rolling_direction == "Easing"

    if near_peak and (pressure_building or rolling_rising or rising_load) and (volatile or backlog_accumulating):
        return {
            "status": "Critical",
            "severity": "critical",
            "status_rank": 6,
            "summary": "System load is near the selected-period peak while pressure and instability signals are active.",
            "leadership_focus": "Prioritize surge-readiness, placement capacity review, and intake-discharge monitoring.",
        }

    if (pressure_building and rolling_rising) or backlog_accumulating:
        return {
            "status": "Pressure",
            "severity": "pressure",
            "status_rank": 5,
            "summary": "Care-load pressure is building through sustained inflow imbalance or rising trend movement.",
            "leadership_focus": "Monitor transfers, discharges, backlog pressure, and near-term capacity buffers.",
        }

    if above_average or elevated_range or rising_load:
        return {
            "status": "Elevated",
            "severity": "elevated",
            "status_rank": 4,
            "summary": "System load is above normal selected-period context or moving upward.",
            "leadership_focus": "Watch whether elevated load persists and prepare operational buffers if trend continues.",
        }

    if watch_volatility or backlog_watch:
        return {
            "status": "Watch",
            "severity": "watch",
            "status_rank": 3,
            "summary": "Early risk signals are present through moderate volatility, backlog pressure, or mixed movement.",
            "leadership_focus": "Track volatility, rolling trend confirmation, and intake-discharge balance.",
        }

    if relief_improving and rolling_easing and falling_load:
        return {
            "status": "Relief",
            "severity": "relief",
            "status_rank": 2,
            "summary": "The selected period shows relief movement with declining load and favorable flow balance.",
            "leadership_focus": "Validate whether relief is sustained before reducing monitoring intensity.",
        }

    return {
        "status": "Stable",
        "severity": "stable",
        "status_rank": 1,
        "summary": "System load appears controlled with no dominant pressure signal in the selected period.",
        "leadership_focus": "Maintain routine monitoring and continue watching for intake shocks or trend shifts.",
    }


def get_module1_status_colors(severity: str) -> dict[str, str]:
    """Return business-meaningful colors for Module 1 status states."""
    severity_key = str(severity or "neutral").strip().lower()

    color_map = {
        "critical": {
            "primary": "#dc2626",
            "secondary": "#991b1b",
            "soft": "#fee2e2",
            "text": "#7f1d1d",
            "accent": "#ef4444",
        },
        "pressure": {
            "primary": "#ea580c",
            "secondary": "#c2410c",
            "soft": "#ffedd5",
            "text": "#7c2d12",
            "accent": "#f97316",
        },
        "elevated": {
            "primary": "#d97706",
            "secondary": "#b45309",
            "soft": "#fef3c7",
            "text": "#78350f",
            "accent": "#f59e0b",
        },
        "watch": {
            "primary": "#ca8a04",
            "secondary": "#a16207",
            "soft": "#fef9c3",
            "text": "#713f12",
            "accent": "#eab308",
        },
        "relief": {
            "primary": "#0f766e",
            "secondary": "#0d9488",
            "soft": "#ccfbf1",
            "text": "#134e4a",
            "accent": "#14b8a6",
        },
        "stable": {
            "primary": "#2563eb",
            "secondary": "#1d4ed8",
            "soft": "#dbeafe",
            "text": "#1e3a8a",
            "accent": "#60a5fa",
        },
        "neutral": {
            "primary": "#64748b",
            "secondary": "#475569",
            "soft": "#f1f5f9",
            "text": "#334155",
            "accent": "#94a3b8",
        },
    }

    return color_map.get(severity_key, color_map["neutral"])


def calculate_recent_average_delta(series: pd.Series) -> float:
    """Estimate recent trend direction when rolling average columns are unavailable."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < 4:
        return 0.0

    split_point = max(1, len(values) // 2)
    prior_avg = safe_float(values.iloc[:split_point].mean())
    recent_avg = safe_float(values.iloc[split_point:].mean())

    return recent_avg - prior_avg


def get_rolling_trend_direction(df: pd.DataFrame) -> dict[str, Any]:
    """Determine whether short-term trend is rising, easing, or stable."""
    if df.empty or "total_system_load" not in df.columns:
        return {
            "direction": "Stable",
            "difference": 0.0,
            "label": "Stable Trend",
            "description": "Trend direction is unavailable for the selected period.",
        }

    latest_7 = latest_value(df, "rolling_7d_avg_system_load") if "rolling_7d_avg_system_load" in df.columns else 0.0
    latest_14 = latest_value(df, "rolling_14d_avg_system_load") if "rolling_14d_avg_system_load" in df.columns else 0.0

    if latest_7 > 0 and latest_14 > 0:
        difference = latest_7 - latest_14
        reference = latest_14
    else:
        difference = calculate_recent_average_delta(df["total_system_load"])
        reference = safe_float(df["total_system_load"].mean())

    threshold = max(reference * 0.01, 1.0)

    if difference > threshold:
        return {
            "direction": "Rising",
            "difference": difference,
            "label": "Rising Trend",
            "description": "Short-term load movement is above the longer trend, indicating pressure buildup.",
        }

    if difference < -threshold:
        return {
            "direction": "Easing",
            "difference": difference,
            "label": "Easing Trend",
            "description": "Short-term load movement is below the longer trend, indicating relief movement.",
        }

    return {
        "direction": "Stable",
        "difference": difference,
        "label": "Stable Trend",
        "description": "Short-term and longer trend signals are closely aligned.",
    }


def calculate_positive_streak(series: pd.Series) -> int:
    """Calculate latest streak of positive net-intake periods."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).tolist()

    streak = 0
    for value in reversed(values):
        if value > 0:
            streak += 1
        else:
            break

    return streak



def get_module1_metric_catalog() -> dict[str, tuple[str, str]]:
    """Return the approved Module 1 diagnostic metric catalogue."""
    return {
        "Total System Load": ("total_system_load", COLORS["navy"]),
        "CBP Custody": ("cbp_custody", COLORS["slate"]),
        "HHS Care": ("hhs_care", COLORS["blue"]),
        "Transfers into HHS": ("transfers_to_hhs", COLORS["amber"]),
        "Discharges from HHS": ("hhs_discharged", COLORS["teal"]),
        "Net Daily Intake": ("net_daily_intake", COLORS["orange"]),
        "Care Load Growth Rate": ("care_load_growth_rate_pct", COLORS["purple"]),
        "Backlog Severity Score": ("backlog_severity_score", COLORS["red"]),
        "Final Pressure Stress Score": ("final_pressure_stress_score", COLORS["green"]),
    }


def identify_module1_dominant_risk_signal(metrics: dict[str, Any]) -> dict[str, Any]:
    """Identify the strongest current Module 1 risk driver for executive interpretation."""
    volatility_index = safe_float(metrics.get("volatility_index"))
    backlog_pressure_share = safe_float(metrics.get("backlog_pressure_share"))
    load_position_pct = safe_float(metrics.get("load_position_pct"))
    discharge_offset_ratio = safe_float(metrics.get("discharge_offset_ratio"))
    positive_intake_streak = safe_float(metrics.get("positive_intake_streak"))

    risk_rows = [
        {
            "label": "Volatility",
            "score": min(max((volatility_index / 8) * 100, 0), 100),
            "value": format_pct(volatility_index, 1),
            "message": "growth volatility is the strongest planning reliability risk",
        },
        {
            "label": "Backlog Pressure",
            "score": min(max(backlog_pressure_share, 0), 100),
            "value": format_pct(backlog_pressure_share, 1),
            "message": "positive net intake appears frequently in the selected period",
        },
        {
            "label": "Load Position",
            "score": min(max(load_position_pct, 0), 100),
            "value": f"{format_pct(load_position_pct, 1)} of peak",
            "message": "current load is positioned close to the selected-period peak",
        },
        {
            "label": "Discharge Offset Gap",
            "score": min(max((1 - min(discharge_offset_ratio, 1)) * 100, 0), 100),
            "value": format_ratio(discharge_offset_ratio),
            "message": "discharges are not fully offsetting transfers",
        },
        {
            "label": "Positive Intake Streak",
            "score": min(max((positive_intake_streak / 7) * 100, 0), 100),
            "value": f"{int(positive_intake_streak)} periods",
            "message": "positive intake pressure is persisting across recent periods",
        },
    ]

    strongest = max(risk_rows, key=lambda row: safe_float(row["score"])) if risk_rows else {
        "label": "No Dominant Risk",
        "score": 0.0,
        "value": "0",
        "message": "no dominant risk signal is active",
    }

    if safe_float(strongest["score"]) < 20:
        return {
            "label": "No Dominant Risk",
            "score": safe_float(strongest["score"]),
            "value": strongest["value"],
            "message": "no dominant risk signal is active; continue routine monitoring",
        }

    return strongest


def calculate_module1_selected_metric_diagnostics(
    df: pd.DataFrame,
    selected_metrics: list[str] | None,
    granularity_context: dict[str, str],
) -> dict[str, Any]:
    """Calculate metric-toggle-aware diagnostics for the selected metric explorer."""
    selected_metrics = selected_metrics or []
    period_label = granularity_context.get("period_label", "daily")
    catalog = get_module1_metric_catalog()

    if df.empty or "date" not in df.columns or not selected_metrics:
        return {
            "selected_count": 0,
            "available_count": 0,
            "strongest_label": "No metric selected",
            "strongest_delta": 0.0,
            "strongest_pct_delta": 0.0,
            "alignment_state": "Unavailable",
            "rising_count": 0,
            "easing_count": 0,
            "stable_count": 0,
            "insight": "No diagnostic metrics are selected. Select sidebar metrics to inspect supporting operational signals.",
            "recommendation": "Use metric toggles as a diagnostic layer after reviewing the executive operating status.",
        }

    available: list[dict[str, Any]] = []

    for metric_label in selected_metrics:
        metric_config = catalog.get(metric_label)
        if not metric_config:
            continue

        column, _ = metric_config
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        first_value = safe_float(series.iloc[0])
        latest_metric_value = safe_float(series.iloc[-1])
        delta = latest_metric_value - first_value
        pct_delta = (delta / abs(first_value) * 100) if abs(first_value) > 0 else 0.0

        if pct_delta > 2:
            direction = "rising"
        elif pct_delta < -2:
            direction = "easing"
        else:
            direction = "stable"

        available.append(
            {
                "label": metric_label,
                "column": column,
                "first": first_value,
                "latest": latest_metric_value,
                "delta": delta,
                "pct_delta": pct_delta,
                "direction": direction,
            }
        )

    if not available:
        return {
            "selected_count": len(selected_metrics),
            "available_count": 0,
            "strongest_label": "Unavailable metrics",
            "strongest_delta": 0.0,
            "strongest_pct_delta": 0.0,
            "alignment_state": "Unavailable",
            "rising_count": 0,
            "easing_count": 0,
            "stable_count": 0,
            "insight": "The selected diagnostic metrics are not available for the current filter state.",
            "recommendation": "Adjust metric toggles or date range before interpreting diagnostic metric movement.",
        }

    strongest = max(available, key=lambda item: abs(safe_float(item["pct_delta"])))
    rising_count = sum(1 for item in available if item["direction"] == "rising")
    easing_count = sum(1 for item in available if item["direction"] == "easing")
    stable_count = sum(1 for item in available if item["direction"] == "stable")

    if rising_count and easing_count:
        alignment_state = "Mixed"
        alignment_text = "selected metrics are moving in mixed directions"
    elif rising_count:
        alignment_state = "Rising"
        alignment_text = "selected metrics are mostly rising"
    elif easing_count:
        alignment_state = "Easing"
        alignment_text = "selected metrics are mostly easing"
    else:
        alignment_state = "Stable"
        alignment_text = "selected metrics are mostly stable"

    strongest_direction = strongest["direction"]
    strongest_direction_text = {
        "rising": "increased",
        "easing": "decreased",
        "stable": "remained broadly stable",
    }.get(strongest_direction, "changed")

    insight = (
        f"{len(available)} selected metric(s) are available in the {period_label} diagnostic view; "
        f"{alignment_text}. The strongest movement is {strongest['label']}, which {strongest_direction_text} "
        f"by {format_pct(strongest['pct_delta'], 1)} from the first to latest selected period."
    )

    if alignment_state == "Mixed":
        recommendation = "Use the diagnostic explorer to identify divergence between care-load, flow, and risk signals before drawing a single conclusion."
    elif alignment_state == "Rising":
        recommendation = "Review whether rising diagnostic metrics reinforce the current operating status or point to emerging pressure."
    elif alignment_state == "Easing":
        recommendation = "Confirm whether easing diagnostic metrics support sustained relief before lowering monitoring intensity."
    else:
        recommendation = "Keep the diagnostic explorer as supporting evidence because selected metrics show limited movement."

    return {
        "selected_count": len(selected_metrics),
        "available_count": len(available),
        "strongest_label": strongest["label"],
        "strongest_delta": safe_float(strongest["delta"]),
        "strongest_pct_delta": safe_float(strongest["pct_delta"]),
        "alignment_state": alignment_state,
        "rising_count": rising_count,
        "easing_count": easing_count,
        "stable_count": stable_count,
        "metrics": available,
        "insight": insight,
        "recommendation": recommendation,
    }


def build_module1_status_reasoning(
    metrics: dict[str, Any],
    status: dict[str, Any],
    granularity_context: dict[str, str],
) -> str:
    """Build a concise dynamic explanation for why the operating status was assigned."""
    status_name = str(status.get("status", "Stable"))
    period_label = granularity_context.get("period_label", "daily")
    load_position_pct = safe_float(metrics.get("load_position_pct"))
    load_delta_pct = safe_float(metrics.get("load_delta_pct"))
    net_intake_pressure = safe_float(metrics.get("net_intake_pressure"))
    volatility_index = safe_float(metrics.get("volatility_index"))
    backlog_pressure_share = safe_float(metrics.get("backlog_pressure_share"))
    rolling_direction = str(metrics.get("rolling_trend_direction", "Stable"))
    dominant_risk = metrics.get("dominant_risk_signal", {}) or {}

    return (
        f"{status_name} is assigned from the current {period_label} view because load is "
        f"{format_pct(load_position_pct, 1)} of peak, latest load movement is {format_pct(load_delta_pct, 1)}, "
        f"net intake averages {format_compact_number(net_intake_pressure, 1)}, volatility is {format_pct(volatility_index, 1)}, "
        f"backlog pressure appears in {format_pct(backlog_pressure_share, 1)} of periods, and the rolling trend is {rolling_direction.lower()}. "
        f"Dominant signal: {dominant_risk.get('label', 'No Dominant Risk')}."
    )


def calculate_module1_intelligence(
    df_visual: pd.DataFrame,
    df_filtered_daily: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Create a complete dynamic intelligence object for Module 1."""
    granularity_context = get_granularity_context(selected_granularity)
    selected_metrics = selected_metrics or []

    if df_visual.empty or "total_system_load" not in df_visual.columns:
        status = classify_module1_status(
            current_load=0.0,
            average_load=0.0,
            peak_load=0.0,
            load_delta_pct=0.0,
            net_intake_pressure=0.0,
            volatility_index=0.0,
            backlog_pressure_share=0.0,
            rolling_direction="Stable",
        )
        colors = get_module1_status_colors(status["severity"])

        return {
            "has_data": False,
            "granularity": selected_granularity,
            "granularity_context": granularity_context,
            "selected_metrics": selected_metrics,
            "status": status,
            "colors": colors,
            "metrics": {},
            "insights": {
                "situation": "No data is available for the selected filter state.",
                "trend": "Trend analysis cannot be completed without system load data.",
                "pressure": "Pressure analysis cannot be completed without intake and discharge data.",
                "capacity": "Capacity position cannot be evaluated for the selected period.",
                "risk": "Risk and stability cannot be evaluated for the selected period.",
            },
            "recommendations": {
                "leadership": "Validate the selected date range and dataset availability.",
                "operations": "Confirm that the required system-load fields are present.",
                "planning": "Avoid operational conclusions until data availability is restored.",
            },
        }

    df_sorted = df_visual.sort_values("date").copy() if "date" in df_visual.columns else df_visual.copy()
    daily_sorted = (
        df_filtered_daily.sort_values("date").copy()
        if not df_filtered_daily.empty and "date" in df_filtered_daily.columns
        else pd.DataFrame()
    )

    current_load = latest_value(df_sorted, "total_system_load")
    previous_load = previous_value(df_sorted, "total_system_load")
    load_delta = current_load - previous_load
    load_delta_pct = (load_delta / previous_load * 100) if previous_load else 0.0

    average_load = safe_float(df_sorted["total_system_load"].mean())
    peak_load = safe_float(df_sorted["total_system_load"].max())
    lowest_load = safe_float(df_sorted["total_system_load"].min())
    load_range = peak_load - lowest_load
    load_position_pct = (current_load / peak_load * 100) if peak_load else 0.0

    if "date" in df_sorted.columns and peak_load > 0:
        peak_rows = df_sorted.loc[df_sorted["total_system_load"] == peak_load]
        peak_date_value = peak_rows["date"].iloc[0] if not peak_rows.empty else None
        peak_date_display = (
            pd.to_datetime(peak_date_value).strftime("%b %d, %Y")
            if peak_date_value is not None and not pd.isna(peak_date_value)
            else "Unavailable"
        )
    else:
        peak_date_display = "Unavailable"

    latest_date_display = "Unavailable"
    if "date" in df_sorted.columns and not df_sorted["date"].dropna().empty:
        latest_date_display = pd.to_datetime(df_sorted["date"].dropna().iloc[-1]).strftime("%b %d, %Y")

    rolling_trend = get_rolling_trend_direction(df_sorted)

    net_intake_pressure = (
        safe_float(df_sorted["net_daily_intake"].mean())
        if "net_daily_intake" in df_sorted.columns
        else 0.0
    )

    total_transfers = (
        safe_float(df_sorted["transfers_to_hhs"].sum())
        if "transfers_to_hhs" in df_sorted.columns
        else 0.0
    )
    total_discharges = (
        safe_float(df_sorted["hhs_discharged"].sum())
        if "hhs_discharged" in df_sorted.columns
        else 0.0
    )
    discharge_offset_ratio = total_discharges / total_transfers if total_transfers > 0 else 0.0

    if "care_load_growth_rate_pct" in df_sorted.columns:
        growth_values = pd.to_numeric(df_sorted["care_load_growth_rate_pct"], errors="coerce").dropna()
        average_growth_rate = safe_float(growth_values.mean()) if not growth_values.empty else 0.0
        volatility_index = safe_float(growth_values.std()) if len(growth_values) > 1 else 0.0
        latest_growth_rate = safe_float(growth_values.iloc[-1]) if not growth_values.empty else 0.0
    else:
        average_growth_rate = 0.0
        volatility_index = 0.0
        latest_growth_rate = 0.0

    if "positive_net_intake_flag" in df_sorted.columns:
        backlog_pressure_share = safe_float(
            pd.to_numeric(df_sorted["positive_net_intake_flag"], errors="coerce").fillna(0).mean() * 100
        )
    elif "net_daily_intake" in df_sorted.columns:
        backlog_pressure_share = safe_float((df_sorted["net_daily_intake"] > 0).mean() * 100)
    else:
        backlog_pressure_share = 0.0

    if "net_daily_intake" in df_sorted.columns:
        positive_intake_streak = calculate_positive_streak(df_sorted["net_daily_intake"])
    else:
        positive_intake_streak = 0

    selected_periods = len(df_sorted)
    selected_days = len(daily_sorted) if not daily_sorted.empty else selected_periods

    status = classify_module1_status(
        current_load=current_load,
        average_load=average_load,
        peak_load=peak_load,
        load_delta_pct=load_delta_pct,
        net_intake_pressure=net_intake_pressure,
        volatility_index=volatility_index,
        backlog_pressure_share=backlog_pressure_share,
        rolling_direction=rolling_trend["direction"],
    )
    colors = get_module1_status_colors(status["severity"])

    metrics = {
        "current_load": current_load,
        "previous_load": previous_load,
        "load_delta": load_delta,
        "load_delta_pct": load_delta_pct,
        "average_load": average_load,
        "peak_load": peak_load,
        "lowest_load": lowest_load,
        "load_range": load_range,
        "load_position_pct": load_position_pct,
        "peak_date_display": peak_date_display,
        "latest_date_display": latest_date_display,
        "rolling_trend_direction": rolling_trend["direction"],
        "rolling_trend_difference": safe_float(rolling_trend["difference"]),
        "rolling_trend_label": rolling_trend["label"],
        "net_intake_pressure": net_intake_pressure,
        "total_transfers": total_transfers,
        "total_discharges": total_discharges,
        "discharge_offset_ratio": discharge_offset_ratio,
        "average_growth_rate": average_growth_rate,
        "latest_growth_rate": latest_growth_rate,
        "volatility_index": volatility_index,
        "backlog_pressure_share": backlog_pressure_share,
        "positive_intake_streak": positive_intake_streak,
        "selected_periods": selected_periods,
        "selected_days": selected_days,
    }

    dominant_risk_signal = identify_module1_dominant_risk_signal(metrics)
    selected_metric_diagnostics = calculate_module1_selected_metric_diagnostics(
        df=df_sorted,
        selected_metrics=selected_metrics,
        granularity_context=granularity_context,
    )

    metrics["dominant_risk_signal"] = dominant_risk_signal
    metrics["selected_metric_diagnostics"] = selected_metric_diagnostics

    insights = generate_module1_insights(metrics, status, granularity_context)
    insights["status_reasoning"] = build_module1_status_reasoning(metrics, status, granularity_context)
    insights["selected_metric"] = selected_metric_diagnostics.get(
        "insight",
        insights.get("selected_metric", "Selected metric diagnostic insight is unavailable."),
    )

    recommendations = generate_module1_recommendations(metrics, status, granularity_context)
    recommendations["diagnostic"] = selected_metric_diagnostics.get(
        "recommendation",
        "Use metric toggles as supporting diagnostic evidence only.",
    )

    return {
        "has_data": True,
        "granularity": selected_granularity,
        "granularity_context": granularity_context,
        "selected_metrics": selected_metrics,
        "status": status,
        "colors": colors,
        "metrics": metrics,
        "insights": insights,
        "recommendations": recommendations,
    }


def generate_module1_insights(
    metrics: dict[str, Any],
    status: dict[str, Any],
    granularity_context: dict[str, str],
) -> dict[str, str]:
    """Generate fully dynamic Module 1 insights from calculated intelligence metrics."""
    current_load = safe_float(metrics.get("current_load"))
    average_load = safe_float(metrics.get("average_load"))
    peak_load = safe_float(metrics.get("peak_load"))
    load_delta = safe_float(metrics.get("load_delta"))
    load_delta_pct = safe_float(metrics.get("load_delta_pct"))
    load_position_pct = safe_float(metrics.get("load_position_pct"))
    net_intake_pressure = safe_float(metrics.get("net_intake_pressure"))
    volatility_index = safe_float(metrics.get("volatility_index"))
    backlog_pressure_share = safe_float(metrics.get("backlog_pressure_share"))
    discharge_offset_ratio = safe_float(metrics.get("discharge_offset_ratio"))
    rolling_direction = str(metrics.get("rolling_trend_direction", "Stable"))
    period_label = granularity_context.get("period_label", "daily")
    movement_label = granularity_context.get("movement_label", "daily movement")
    average_label = granularity_context.get("average_label", "average per day")

    if peak_load <= 0:
        return {
            "situation": "System load cannot be interpreted because peak load is unavailable.",
            "trend": "Trend direction cannot be evaluated without enough system-load data.",
            "pressure": "Pressure balance cannot be evaluated without transfer and discharge flow.",
            "capacity": "Capacity position cannot be evaluated for the selected period.",
            "risk": "Risk and stability signals are unavailable for the selected period.",
            "selected_metric": "Selected metric behavior will be available when metric data is present.",
        }

    if load_position_pct >= 90:
        situation = (
            f"The latest system load is operating near the selected-period peak "
            f"({format_pct(load_position_pct, 1)} of peak load), indicating high capacity attention."
        )
    elif current_load >= average_load:
        situation = (
            f"The latest system load is above the selected-period average, showing elevated care responsibility "
            f"for the current {period_label} view."
        )
    elif load_delta < 0:
        situation = (
            f"The latest system load is below the selected-period average and has moved downward, "
            f"suggesting relief in the current {period_label} view."
        )
    else:
        situation = (
            f"The latest system load is within a controlled range compared with the selected-period average."
        )

    if rolling_direction == "Rising":
        trend = (
            f"The rolling trend is rising, meaning short-term {movement_label} is above the longer trend "
            f"and may signal renewed care-load pressure."
        )
    elif rolling_direction == "Easing":
        trend = (
            f"The rolling trend is easing, meaning recent {movement_label} is moving below the longer trend "
            f"and may indicate improving system relief."
        )
    else:
        trend = (
            f"The rolling trend is stable, meaning recent and longer-term care-load movement are closely aligned."
        )

    if net_intake_pressure > 0:
        pressure = (
            f"Net intake pressure is positive at {format_compact_number(net_intake_pressure, 1)} {average_label}, "
            f"which means transfers into HHS are exceeding discharges on average."
        )
    elif net_intake_pressure < 0:
        pressure = (
            f"Net intake pressure is negative at {format_compact_number(net_intake_pressure, 1)} {average_label}, "
            f"which means discharges are offsetting incoming transfers during the selected period."
        )
    else:
        pressure = (
            f"Transfers and discharges are nearly balanced, limiting additional intake pressure."
        )

    if load_position_pct >= 90:
        capacity = (
            f"Current load is close to peak range, leaving limited room before the highest selected-period burden."
        )
    elif load_position_pct >= 75:
        capacity = (
            f"Current load is in an elevated range at {format_pct(load_position_pct, 1)} of selected-period peak."
        )
    elif load_position_pct >= 50:
        capacity = (
            f"Current load is in a moderate range compared with the selected-period peak."
        )
    else:
        capacity = (
            f"Current load is closer to the relief range than the peak range for the selected period."
        )

    if volatility_index > 5 or backlog_pressure_share > 60:
        risk = (
            f"Risk signals require attention: volatility is {format_pct(volatility_index, 1)} and backlog pressure "
            f"appears in {format_pct(backlog_pressure_share, 1)} of selected periods."
        )
    elif volatility_index > 2 or backlog_pressure_share > 40:
        risk = (
            f"Moderate risk signals are present, with volatility at {format_pct(volatility_index, 1)} and "
            f"backlog pressure share at {format_pct(backlog_pressure_share, 1)}."
        )
    else:
        risk = (
            f"Risk signals are controlled, with volatility at {format_pct(volatility_index, 1)} and backlog pressure "
            f"share at {format_pct(backlog_pressure_share, 1)}."
        )

    if discharge_offset_ratio >= 1:
        selected_metric = (
            f"Discharge offset is favorable at {format_ratio(discharge_offset_ratio)}, supporting relief capacity."
        )
    elif discharge_offset_ratio >= 0.85:
        selected_metric = (
            f"Discharge offset is near balance at {format_ratio(discharge_offset_ratio)}, requiring continued monitoring."
        )
    else:
        selected_metric = (
            f"Discharge offset is below balance at {format_ratio(discharge_offset_ratio)}, indicating potential relief constraints."
        )

    return {
        "situation": situation,
        "trend": trend,
        "pressure": pressure,
        "capacity": capacity,
        "risk": risk,
        "selected_metric": selected_metric,
        "executive_summary": status.get("summary", "Operational status has been calculated from selected-period data."),
    }


def generate_module1_recommendations(
    metrics: dict[str, Any],
    status: dict[str, Any],
    granularity_context: dict[str, str],
) -> dict[str, str]:
    """Generate dynamic recommendations from Module 1 status and metrics."""
    status_name = str(status.get("status", "Stable"))
    net_intake_pressure = safe_float(metrics.get("net_intake_pressure"))
    load_position_pct = safe_float(metrics.get("load_position_pct"))
    volatility_index = safe_float(metrics.get("volatility_index"))
    backlog_pressure_share = safe_float(metrics.get("backlog_pressure_share"))
    discharge_offset_ratio = safe_float(metrics.get("discharge_offset_ratio"))
    period_label = granularity_context.get("period_label", "daily")

    if status_name == "Critical":
        leadership = "Escalate surge-readiness review because load is near peak while pressure or instability signals are active."
        operations = "Monitor intake, transfers, discharges, and placement throughput closely for the selected period."
        planning = "Prepare staffing and shelter capacity buffers until pressure and volatility indicators decline."

    elif status_name == "Pressure":
        leadership = "Prioritize intake-discharge balance because sustained pressure is building across the selected period."
        operations = "Investigate whether transfers are consistently exceeding discharges and monitor backlog accumulation."
        planning = "Prepare short-term operational buffers if pressure remains positive in the next reporting periods."

    elif status_name == "Elevated":
        leadership = "Watch whether elevated system load persists above the selected-period average."
        operations = "Compare current load against peak and rolling trend movement to confirm whether this is temporary or sustained."
        planning = "Maintain capacity flexibility until load returns closer to normal operating range."

    elif status_name == "Watch":
        leadership = "Treat the selected period as an early-monitoring condition rather than a fully stable state."
        operations = "Track volatility, backlog share, and rolling trend direction for confirmation of pressure or relief."
        planning = "Avoid overreacting to one movement period; wait for trend confirmation while maintaining readiness."

    elif status_name == "Relief":
        leadership = "Validate whether relief is sustained before reducing monitoring intensity."
        operations = "Confirm that discharges and placement flow continue to offset incoming transfers."
        planning = "Maintain monitoring coverage because relief can reverse if intake rises again."

    elif status_name == "No Data":
        leadership = "Validate data availability before making operational decisions."
        operations = "Check date range, required fields, and data loading status."
        planning = "Do not use this filter state for planning until data is restored."

    else:
        leadership = "Maintain routine system-load monitoring while watching for intake shocks."
        operations = "Continue tracking transfers, discharges, and rolling load movement."
        planning = "Keep standard staffing and shelter planning assumptions active unless trend conditions change."

    if load_position_pct >= 90:
        risk_watch = "Current load is near selected-period peak; monitor capacity strain and resource readiness."
    elif volatility_index > 5:
        risk_watch = "High volatility may reduce planning reliability; avoid relying only on average load."
    elif backlog_pressure_share > 60:
        risk_watch = "Backlog pressure is persistent; monitor whether positive net intake continues."
    elif discharge_offset_ratio < 0.85 and net_intake_pressure > 0:
        risk_watch = "Discharge offset is below balance while intake pressure is positive; relief capacity may be constrained."
    else:
        risk_watch = "No dominant risk trigger is active, but routine monitoring should continue."

    return {
        "leadership": leadership,
        "operations": operations,
        "planning": planning,
        "risk_watch": risk_watch,
        "period_note": f"All recommendations reflect the current {period_label} view and selected date range.",
    }


# ============================================================
# 8. Plotly Foundation
# ============================================================

def apply_plotly_layout(
    fig: go.Figure,
    title: str,
    height: int = STANDARD_HEIGHT,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
) -> go.Figure:
    """Apply a consistent healthcare analytics Plotly layout."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 18, "color": COLORS["navy"], "family": "Inter, Segoe UI, sans-serif"},
        },
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": COLORS["navy"], "size": 12},
        hovermode="x unified",
        margin={"l": 40, "r": 28, "t": 72, "b": 46},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        },
        xaxis={
            "title": xaxis_title,
            "showgrid": False,
            "linecolor": COLORS["slate_light"],
            "tickfont": {"size": 11, "color": COLORS["slate"]},
        },
        yaxis={
            "title": yaxis_title,
            "gridcolor": "rgba(148, 163, 184, 0.22)",
            "zerolinecolor": "rgba(100, 116, 139, 0.38)",
            "linecolor": COLORS["slate_light"],
            "tickfont": {"size": 11, "color": COLORS["slate"]},
        },
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    """Return a clear empty-state Plotly figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": COLORS["slate"]},
    )
    fig.update_layout(
        height=COMPACT_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def pressure_level_color(level: Any) -> str:
    """Return a dynamic color for pressure/stress levels."""
    normalized = normalize_text(level)

    for key, color in PRESSURE_COLORS.items():
        if key in normalized:
            return color

    return COLORS["slate"]


# ============================================================
# 9. Plotly Visualizations — Module 1
# System Load Overview Command Center Visual Functions
# Patch 2: Dynamic Visual Function Upgrade
# ============================================================

def get_module1_intelligence_value(
    intelligence: dict[str, Any] | None,
    key: str,
    default: Any = None,
) -> Any:
    """Safely fetch a metric/status/color value from the Module 1 intelligence object."""
    if not intelligence:
        return default

    metrics = intelligence.get("metrics", {})
    status = intelligence.get("status", {})
    colors = intelligence.get("colors", {})

    if key in metrics:
        return metrics.get(key, default)

    if key in status:
        return status.get(key, default)

    if key in colors:
        return colors.get(key, default)

    return intelligence.get(key, default)


def get_module1_chart_context(
    intelligence: dict[str, Any] | None,
    fallback_status: str = "Stable",
) -> dict[str, str]:
    """Return dynamic chart context from Module 1 intelligence."""
    if not intelligence:
        return {
            "status": fallback_status,
            "severity": "stable",
            "primary": COLORS["blue"],
            "secondary": COLORS["teal"],
            "accent": COLORS["blue_soft"],
            "soft": "rgba(219, 234, 254, 0.22)",
            "period_label": "daily",
            "movement_label": "daily movement",
            "previous_label": "vs prior day",
        }

    status = intelligence.get("status", {})
    colors = intelligence.get("colors", {})
    context = intelligence.get("granularity_context", {})

    return {
        "status": str(status.get("status", fallback_status)),
        "severity": str(status.get("severity", "stable")),
        "primary": str(colors.get("primary", COLORS["blue"])),
        "secondary": str(colors.get("secondary", COLORS["teal"])),
        "accent": str(colors.get("accent", COLORS["blue_soft"])),
        "soft": str(colors.get("soft", "rgba(219, 234, 254, 0.22)")),
        "period_label": str(context.get("period_label", "daily")),
        "movement_label": str(context.get("movement_label", "daily movement")),
        "previous_label": str(context.get("previous_label", "vs prior day")),
    }



def apply_module1_chart_layout(
    fig: go.Figure,
    title: str,
    height: int,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
    show_legend: bool = True,
    legend_bottom: bool = True,
) -> go.Figure:
    """Apply a Module 1-specific Plotly layout that prevents title, legend, and annotation collisions."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
            "font": {
                "size": 17,
                "color": COLORS["navy"],
                "family": "Inter, Segoe UI, sans-serif",
            },
        },
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        hovermode="x unified",
        margin={
            "l": 58,
            "r": 72,
            "t": 72,
            "b": 86 if show_legend and legend_bottom else 54,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16 if legend_bottom else 1.08,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10, "color": COLORS["navy"]},
            "itemwidth": 30,
        },
        showlegend=show_legend,
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": COLORS["navy"],
        },
    )

    fig.update_xaxes(
        title_text=xaxis_title,
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(148, 163, 184, 0.55)",
        tickfont={"size": 11, "color": COLORS["slate"]},
        rangeslider={"visible": False},
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(100, 116, 139, 0.35)",
        spikethickness=1,
    )

    fig.update_yaxes(
        title_text=yaxis_title,
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(100, 116, 139, 0.35)",
        tickfont={"size": 11, "color": COLORS["slate"]},
        title_font={"size": 12, "color": COLORS["slate"]},
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(100, 116, 139, 0.28)",
        spikethickness=1,
    )

    return fig

def add_module1_peak_and_latest_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    y_column: str,
    color_primary: str,
    color_accent: str,
) -> go.Figure:
    """Add dynamic peak and latest markers to a Module 1 trend figure."""
    if df.empty or "date" not in df.columns or y_column not in df.columns:
        return fig

    clean_df = df[["date", y_column]].dropna().copy()

    if clean_df.empty:
        return fig

    peak_idx = clean_df[y_column].idxmax()
    latest_idx = clean_df.index[-1]

    peak_date = clean_df.loc[peak_idx, "date"]
    peak_value = safe_float(clean_df.loc[peak_idx, y_column])

    latest_date = clean_df.loc[latest_idx, "date"]
    latest_value_number = safe_float(clean_df.loc[latest_idx, y_column])

    fig.add_trace(
        go.Scatter(
            x=[peak_date],
            y=[peak_value],
            mode="markers+text",
            name="Selected-Period Peak",
            marker={
                "size": 12,
                "color": color_primary,
                "line": {"width": 2, "color": "#ffffff"},
                "symbol": "diamond",
            },
            text=["Peak"],
            textposition="top center",
            textfont={"size": 11, "color": color_primary},
            hovertemplate="%{x|%b %d, %Y}<br>Peak Load: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_value_number],
            mode="markers+text",
            name="Latest Load",
            marker={
                "size": 12,
                "color": color_accent,
                "line": {"width": 2, "color": "#ffffff"},
                "symbol": "circle",
            },
            text=["Latest"],
            textposition="bottom center",
            textfont={"size": 11, "color": color_accent},
            hovertemplate="%{x|%b %d, %Y}<br>Latest Load: %{y:,.0f}<extra></extra>",
        )
    )

    return fig



def create_system_load_trend(
    df: pd.DataFrame,
    intelligence: dict[str, Any] | None = None,
    selected_granularity: str = "Daily",
) -> go.Figure:
    """Create the primary executive system-load trend chart without title/legend collisions."""
    required = {"date", "total_system_load"}

    if df.empty or not required.issubset(df.columns):
        return empty_figure("System load trend is unavailable because required columns are missing.")

    chart_context = get_module1_chart_context(intelligence)
    status_label = chart_context["status"]
    primary_color = chart_context["primary"]
    secondary_color = chart_context["secondary"]
    accent_color = chart_context["accent"]

    current_load = safe_float(get_module1_intelligence_value(intelligence, "current_load", latest_value(df, "total_system_load")))
    average_load = safe_float(get_module1_intelligence_value(intelligence, "average_load", df["total_system_load"].mean()))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["total_system_load"],
            mode="lines",
            name="Total Load",
            line={"color": primary_color, "width": 3.5},
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.065)",
            hovertemplate="%{x|%b %d, %Y}<br>Total system load: %{y:,.0f}<extra></extra>",
        )
    )

    if "rolling_7d_avg_system_load" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rolling_7d_avg_system_load"],
                mode="lines",
                name="Short rolling avg",
                line={"color": accent_color, "width": 2.4},
                hovertemplate="%{x|%b %d, %Y}<br>Short rolling avg: %{y:,.0f}<extra></extra>",
            )
        )

    if "rolling_14d_avg_system_load" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rolling_14d_avg_system_load"],
                mode="lines",
                name="Long rolling avg",
                line={"color": secondary_color, "width": 2.2, "dash": "dot"},
                hovertemplate="%{x|%b %d, %Y}<br>Long rolling avg: %{y:,.0f}<extra></extra>",
            )
        )

    fig = add_module1_peak_and_latest_markers(
        fig=fig,
        df=df,
        y_column="total_system_load",
        color_primary=primary_color,
        color_accent=accent_color,
    )

    if average_load > 0:
        fig.add_hline(
            y=average_load,
            line_dash="dash",
            line_color="rgba(100, 116, 139, 0.52)",
            annotation_text=f"Avg {format_compact_number(average_load)}",
            annotation_position="bottom right",
        )

    fig = apply_module1_chart_layout(
        fig,
        title=f"System Load Trend — {status_label}",
        yaxis_title="Children Under Care",
        height=STANDARD_HEIGHT + 40,
        show_legend=True,
        legend_bottom=True,
    )

    fig.update_traces(cliponaxis=False)

    return fig


def create_growth_rate_trend(
    df: pd.DataFrame,
    intelligence: dict[str, Any] | None = None,
    selected_granularity: str = "Daily",
) -> go.Figure:
    """Create dynamic pressure vs relief movement chart from care load growth rate."""
    if df.empty or "care_load_growth_rate_pct" not in df.columns or "date" not in df.columns:
        return empty_figure("Care load growth rate trend is unavailable because required columns are missing.")

    chart_context = get_module1_chart_context(intelligence)
    status_label = chart_context["status"]
    pressure_color = chart_context["primary"] if chart_context["severity"] in {"critical", "pressure", "elevated", "watch"} else COLORS["orange"]
    relief_color = COLORS["teal"]
    stable_color = "rgba(100, 116, 139, 0.52)"
    movement_label = chart_context["movement_label"]

    plot_df = df[["date", "care_load_growth_rate_pct"]].copy()
    plot_df["growth_rate"] = pd.to_numeric(plot_df["care_load_growth_rate_pct"], errors="coerce").fillna(0.0)

    neutral_threshold = 0.05
    plot_df["movement_state"] = np.select(
        [
            plot_df["growth_rate"] > neutral_threshold,
            plot_df["growth_rate"] < -neutral_threshold,
        ],
        ["Pressure Movement", "Relief Movement"],
        default="Balanced Movement",
    )

    plot_df["bar_color"] = np.select(
        [
            plot_df["movement_state"] == "Pressure Movement",
            plot_df["movement_state"] == "Relief Movement",
        ],
        [pressure_color, relief_color],
        default=stable_color,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["date"],
            y=plot_df["growth_rate"],
            name="Movement",
            marker={
                "color": plot_df["bar_color"],
                "line": {"width": 0},
            },
            hovertemplate=(
                "%{x|%b %d, %Y}"
                "<br>Growth rate: %{y:.2f}%"
                "<br>Signal: %{customdata}<extra></extra>"
            ),
            customdata=plot_df["movement_state"],
        )
    )

    average_growth = safe_float(get_module1_intelligence_value(intelligence, "average_growth_rate", plot_df["growth_rate"].mean()))
    latest_growth = safe_float(get_module1_intelligence_value(intelligence, "latest_growth_rate", plot_df["growth_rate"].iloc[-1] if not plot_df.empty else 0.0))
    volatility_index = safe_float(get_module1_intelligence_value(intelligence, "volatility_index", plot_df["growth_rate"].std()))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(71, 85, 105, 0.70)",
        annotation_text="Balanced baseline",
        annotation_position="top left",
    )

    if len(plot_df) >= 2:
        fig.add_hline(
            y=average_growth,
            line_dash="dot",
            line_color="rgba(15, 23, 42, 0.42)",
            annotation_text=f"Avg {format_pct(average_growth, 2)}",
            annotation_position="bottom right",
        )

    fig = apply_module1_chart_layout(
        fig,
        title=f"Pressure vs Relief Movement — {status_label}",
        height=COMPACT_HEIGHT + 45,
        yaxis_title="Growth Rate (%)",
        show_legend=False,
    )

    fig.update_layout(
        margin={"l": 58, "r": 62, "t": 72, "b": 54},
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=1.08,
        xanchor="right",
        showarrow=False,
        align="right",
        text=f"Latest {format_pct(latest_growth, 2)} · Volatility {format_pct(volatility_index, 1)} · {movement_label.title()}",
        font={"size": 11, "color": COLORS["slate"]},
    )

    return fig


def create_capacity_position_view(
    intelligence: dict[str, Any] | None,
) -> go.Figure:
    """Create a dynamic capacity position visual showing latest load versus low, average, and peak range."""
    if not intelligence or not intelligence.get("has_data", False):
        return empty_figure("Capacity position view is unavailable because Module 1 intelligence data is missing.")

    metrics = intelligence.get("metrics", {})
    chart_context = get_module1_chart_context(intelligence)

    current_load = safe_float(metrics.get("current_load"))
    average_load = safe_float(metrics.get("average_load"))
    peak_load = safe_float(metrics.get("peak_load"))
    lowest_load = safe_float(metrics.get("lowest_load"))
    load_position_pct = safe_float(metrics.get("load_position_pct"))

    if peak_load <= 0:
        return empty_figure("Capacity position cannot be calculated because peak load is unavailable.")

    category_values = [
        {"label": "Lowest Load", "value": lowest_load, "color": COLORS["teal"]},
        {"label": "Current Load", "value": current_load, "color": chart_context["primary"]},
        {"label": "Average Load", "value": average_load, "color": COLORS["blue"]},
        {"label": "Peak Load", "value": peak_load, "color": COLORS["red"] if load_position_pct >= 90 else COLORS["orange"]},
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[item["value"] for item in category_values],
            y=[item["label"] for item in category_values],
            orientation="h",
            marker={
                "color": [item["color"] for item in category_values],
                "line": {"width": 1, "color": "rgba(255,255,255,0.92)"},
            },
            text=[format_compact_number(item["value"]) for item in category_values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Load: %{x:,.0f}<extra></extra>",
            name="Capacity Position",
        )
    )

    fig.add_vline(
        x=current_load,
        line_dash="solid",
        line_color=chart_context["primary"],
    )

    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": ["Peak Load", "Average Load", "Current Load", "Lowest Load"]},
        xaxis={"range": [0, peak_load * 1.16]},
    )

    fig = apply_module1_chart_layout(
        fig,
        title=f"Capacity Position — {chart_context['status']}",
        height=COMPACT_HEIGHT + 30,
        xaxis_title="Children Under Care",
        show_legend=False,
    )

    fig.update_layout(
        margin={"l": 118, "r": 76, "t": 72, "b": 58},
    )

    return fig


def create_risk_stability_signal(
    intelligence: dict[str, Any] | None,
) -> go.Figure:
    """Create compact risk and stability signal visualization for Module 1."""
    if not intelligence or not intelligence.get("has_data", False):
        return empty_figure("Risk and stability signal is unavailable because Module 1 intelligence data is missing.")

    metrics = intelligence.get("metrics", {})
    chart_context = get_module1_chart_context(intelligence)

    volatility_index = safe_float(metrics.get("volatility_index"))
    backlog_pressure_share = safe_float(metrics.get("backlog_pressure_share"))
    load_position_pct = safe_float(metrics.get("load_position_pct"))
    discharge_offset_ratio = safe_float(metrics.get("discharge_offset_ratio"))
    positive_intake_streak = safe_float(metrics.get("positive_intake_streak"))

    risk_rows = [
        {
            "signal": "Volatility",
            "score": min(max((volatility_index / 8) * 100, 0), 100),
            "raw": format_pct(volatility_index, 1),
        },
        {
            "signal": "Backlog Pressure",
            "score": min(max(backlog_pressure_share, 0), 100),
            "raw": format_pct(backlog_pressure_share, 1),
        },
        {
            "signal": "Load Position",
            "score": min(max(load_position_pct, 0), 100),
            "raw": f"{format_pct(load_position_pct, 1)} of peak",
        },
        {
            "signal": "Positive Intake Streak",
            "score": min(max((positive_intake_streak / 7) * 100, 0), 100),
            "raw": f"{int(positive_intake_streak)} periods",
        },
        {
            "signal": "Discharge Offset Gap",
            "score": min(max((1 - min(discharge_offset_ratio, 1)) * 100, 0), 100),
            "raw": format_ratio(discharge_offset_ratio),
        },
    ]

    def risk_color(score: float) -> str:
        if score >= 80:
            return COLORS["red"]
        if score >= 60:
            return COLORS["orange"]
        if score >= 40:
            return COLORS["amber"]
        if score >= 20:
            return COLORS["blue"]
        return COLORS["teal"]

    fig = go.Figure()

    fig.add_vrect(x0=0, x1=40, fillcolor="rgba(20, 184, 166, 0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=40, x1=70, fillcolor="rgba(245, 158, 11, 0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=70, x1=100, fillcolor="rgba(220, 38, 38, 0.07)", line_width=0, layer="below")

    fig.add_trace(
        go.Bar(
            x=[row["score"] for row in risk_rows],
            y=[row["signal"] for row in risk_rows],
            orientation="h",
            marker={
                "color": [risk_color(row["score"]) for row in risk_rows],
                "line": {"width": 1, "color": "rgba(255,255,255,0.90)"},
            },
            text=[row["raw"] for row in risk_rows],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Risk score: %{x:.1f}/100<br>Raw value: %{text}<extra></extra>",
            name="Risk Signal",
        )
    )

    fig.update_layout(
        xaxis={"range": [0, 122]},
        yaxis={"categoryorder": "total ascending"},
    )

    fig = apply_module1_chart_layout(
        fig,
        title=f"Risk & Stability — {chart_context['status']}",
        height=COMPACT_HEIGHT + 30,
        xaxis_title="Normalized Risk Signal",
        show_legend=False,
    )

    fig.update_layout(
        margin={"l": 132, "r": 92, "t": 72, "b": 58},
    )

    return fig


def create_module1_selected_metric_explorer(
    df: pd.DataFrame,
    selected_metrics: list[str],
    intelligence: dict[str, Any] | None = None,
) -> go.Figure:
    """Create optional Module 1 selected metric explorer with dynamic chart context."""
    metric_map = get_module1_metric_catalog()

    if df.empty or not selected_metrics:
        return empty_figure("Select at least one metric from the sidebar to view dynamic metric trends.")

    chart_context = get_module1_chart_context(intelligence)
    diagnostics = {}
    if intelligence:
        diagnostics = intelligence.get("metrics", {}).get("selected_metric_diagnostics", {}) or {}

    fig = go.Figure()
    valid_metrics = []

    for metric_label in selected_metrics:
        config = metric_map.get(metric_label)

        if not config:
            continue

        column, base_color = config

        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")

        if series.dropna().empty:
            continue

        first_value = safe_float(series.dropna().iloc[0])
        latest_metric_value = safe_float(series.dropna().iloc[-1])
        metric_delta = latest_metric_value - first_value

        line_color = chart_context["primary"] if metric_label == "Total System Load" else base_color

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=series,
                mode="lines",
                name=metric_label,
                line={"color": line_color, "width": 2.35},
                hovertemplate=f"%{{x|%b %d, %Y}}<br>{metric_label}: %{{y:,.2f}}<extra></extra>",
            )
        )

        valid_metrics.append(
            {
                "label": metric_label,
                "latest": latest_metric_value,
                "delta": metric_delta,
            }
        )

    if not valid_metrics:
        return empty_figure("Selected metrics are unavailable for the filtered dataset.")

    metric_count = len(valid_metrics)
    alignment_state = diagnostics.get("alignment_state", "Diagnostic")
    strongest_label = diagnostics.get("strongest_label", "Selected metric")

    fig = apply_module1_chart_layout(
        fig,
        title=f"Diagnostic Metric Explorer — {alignment_state} Signals",
        height=COMPACT_HEIGHT + 55,
        yaxis_title="Selected Metric Value",
        show_legend=True,
        legend_bottom=True,
    )

    fig.update_layout(
        margin={"l": 62, "r": 54, "t": 72, "b": 96},
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=1.08,
        xanchor="right",
        showarrow=False,
        align="right",
        text=f"{metric_count} metric(s) visible · strongest: {hhs_escape_html(str(strongest_label))}",
        font={"size": 11, "color": COLORS["slate"]},
    )

    return fig


# ============================================================
# 10. Plotly Visualizations — Module 2
# ============================================================

def create_cbp_hhs_comparison(df: pd.DataFrame) -> go.Figure:
    """Create CBP vs HHS load comparison line chart."""
    required = {"date", "cbp_custody", "hhs_care"}

    if df.empty or not required.issubset(df.columns):
        return empty_figure("CBP vs HHS comparison is unavailable because required columns are missing.")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cbp_custody"],
            mode="lines",
            name="CBP Custody",
            line={"color": COLORS["slate"], "width": 3},
            hovertemplate="%{x|%b %d, %Y}<br>CBP Custody: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["hhs_care"],
            mode="lines",
            name="HHS Care",
            line={"color": COLORS["blue"], "width": 3},
            hovertemplate="%{x|%b %d, %Y}<br>HHS Care: %{y:,.0f}<extra></extra>",
        )
    )

    return apply_plotly_layout(
        fig,
        title="CBP Custody vs HHS Care Load Comparison",
        yaxis_title="Children",
    )


def create_load_composition(df: pd.DataFrame) -> go.Figure:
    """Create stacked area composition for CBP and HHS load."""
    required = {"date", "cbp_custody", "hhs_care"}

    if df.empty or not required.issubset(df.columns):
        return empty_figure("Care load composition is unavailable because required columns are missing.")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cbp_custody"],
            mode="lines",
            stackgroup="one",
            name="CBP Custody",
            line={"color": COLORS["slate"], "width": 1},
            fillcolor="rgba(100, 116, 139, 0.28)",
            hovertemplate="%{x|%b %d, %Y}<br>CBP Custody: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["hhs_care"],
            mode="lines",
            stackgroup="one",
            name="HHS Care",
            line={"color": COLORS["blue"], "width": 1},
            fillcolor="rgba(37, 99, 235, 0.28)",
            hovertemplate="%{x|%b %d, %Y}<br>HHS Care: %{y:,.0f}<extra></extra>",
        )
    )

    return apply_plotly_layout(
        fig,
        title="Care Load Composition — CBP and HHS Share Over Time",
        height=COMPACT_HEIGHT,
        yaxis_title="Children",
    )


def create_latest_load_share_donut(df: pd.DataFrame) -> go.Figure:
    """Create latest CBP/HHS load share donut chart."""
    required = {"cbp_custody", "hhs_care"}

    if df.empty or not required.issubset(df.columns):
        return empty_figure("Latest care load share is unavailable because required columns are missing.")

    latest_df = df.sort_values("date").tail(1)
    cbp_value = latest_value(latest_df, "cbp_custody")
    hhs_value = latest_value(latest_df, "hhs_care")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["CBP Custody", "HHS Care"],
                values=[cbp_value, hhs_value],
                hole=0.64,
                marker={"colors": [COLORS["slate"], COLORS["blue"]]},
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        annotations=[
            {
                "text": "Latest<br>Load Share",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": COLORS["navy"]},
            }
        ],
        showlegend=False,
    )

    return apply_plotly_layout(
        fig,
        title="Latest CBP vs HHS Load Share",
        height=COMPACT_HEIGHT,
    )


# ============================================================
# 11. Plotly Visualizations — Module 3
# ============================================================

def create_transfers_vs_discharges(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart for transfers vs discharges."""
    required = {"date", "transfers_to_hhs", "hhs_discharged"}

    if df.empty or not required.issubset(df.columns):
        return empty_figure("Transfers vs discharges chart is unavailable because required columns are missing.")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["transfers_to_hhs"],
            name="Transfers into HHS",
            marker={"color": COLORS["amber"]},
            hovertemplate="%{x|%b %d, %Y}<br>Transfers: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["hhs_discharged"],
            name="Discharges from HHS",
            marker={"color": COLORS["teal"]},
            hovertemplate="%{x|%b %d, %Y}<br>Discharges: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")

    return apply_plotly_layout(
        fig,
        title="Transfers into HHS vs Discharges from HHS",
        yaxis_title="Children",
    )


def create_net_intake_pressure(df: pd.DataFrame) -> go.Figure:
    """Create net daily intake pressure chart with pressure/relief colors."""
    if df.empty or "net_daily_intake" not in df.columns:
        return empty_figure("Net intake pressure chart is unavailable because required columns are missing.")

    net_intake = df["net_daily_intake"].fillna(0)
    colors = np.where(net_intake >= 0, COLORS["orange"], COLORS["teal"])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=net_intake,
            name="Net Intake Pressure",
            marker={"color": colors},
            hovertemplate="%{x|%b %d, %Y}<br>Net Intake: %{y:,.0f}<extra></extra>",
        )
    )

    if "rolling_7d_avg_net_intake" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rolling_7d_avg_net_intake"],
                mode="lines",
                name="7-Period Net Intake Avg",
                line={"color": COLORS["navy"], "width": 2},
                hovertemplate="%{x|%b %d, %Y}<br>Rolling Avg: %{y:,.1f}<extra></extra>",
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS["slate"],
        annotation_text="Balanced flow",
        annotation_position="top left",
    )

    return apply_plotly_layout(
        fig,
        title="Net Intake Pressure — Inflow vs Outflow Imbalance",
        yaxis_title="Transfers − Discharges",
    )


def create_backlog_severity_trend(df: pd.DataFrame) -> go.Figure:
    """Create backlog severity trend chart."""
    if df.empty or "backlog_severity_score" not in df.columns:
        return empty_figure("Backlog severity trend is unavailable because required columns are missing.")

    severity = df["backlog_severity_score"].fillna(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=severity,
            mode="lines+markers",
            name="Backlog Severity Score",
            line={"color": COLORS["red"], "width": 2},
            marker={"size": 5, "color": COLORS["orange"]},
            fill="tozeroy",
            fillcolor="rgba(249, 115, 22, 0.16)",
            hovertemplate="%{x|%b %d, %Y}<br>Backlog Severity: %{y:.2f}<extra></extra>",
        )
    )

    return apply_plotly_layout(
        fig,
        title="Backlog Severity Trend",
        height=COMPACT_HEIGHT,
        yaxis_title="Severity Score",
    )


def create_pressure_relief_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create compact pressure/relief period heatmap strip."""
    if df.empty or "net_daily_intake" not in df.columns:
        return empty_figure("Pressure/relief calendar strip is unavailable because required columns are missing.")

    heat_df = df[["date", "net_daily_intake"]].copy()
    heat_df["pressure_signal"] = np.select(
        [
            heat_df["net_daily_intake"] > 0,
            heat_df["net_daily_intake"] < 0,
        ],
        [1, -1],
        default=0,
    )

    fig = go.Figure(
        data=go.Heatmap(
            x=heat_df["date"],
            y=["Pressure Signal"] * len(heat_df),
            z=[heat_df["pressure_signal"]],
            colorscale=[
                [0.00, COLORS["teal"]],
                [0.49, COLORS["teal"]],
                [0.50, COLORS["slate_light"]],
                [0.51, COLORS["orange"]],
                [1.00, COLORS["red"]],
            ],
            showscale=True,
            colorbar={
                "title": "Signal",
                "tickvals": [-1, 0, 1],
                "ticktext": ["Relief", "Balanced", "Pressure"],
            },
            hovertemplate="%{x|%b %d, %Y}<br>Signal: %{z}<extra></extra>",
        )
    )

    return apply_plotly_layout(
        fig,
        title="Pressure and Relief Period Strip",
        height=250,
    )


# ============================================================
# 12. Plotly Visualizations — Module 4
# ============================================================

def create_kpi_performance_bar(kpis: dict[str, dict[str, Any]]) -> go.Figure:
    """Create KPI performance horizontal bar chart."""
    labels = [item["label"] for item in kpis.values()]
    values = [safe_float(item["value"]) for item in kpis.values()]
    colors = [
        COLORS["navy"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["red"],
        COLORS["teal"],
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker={"color": colors},
                hovertemplate="%{y}<br>Value: %{x:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": labels[::-1]})

    return apply_plotly_layout(
        fig,
        title="Guideline KPI Performance Summary",
        height=COMPACT_HEIGHT,
        xaxis_title="Dynamic KPI Value",
    )


def create_pressure_distribution(df: pd.DataFrame) -> go.Figure:
    """Create pressure/stress level distribution chart."""
    if df.empty or "final_pressure_stress_level" not in df.columns:
        return empty_figure("Pressure stress level distribution is unavailable because required columns are missing.")

    distribution = (
        df["final_pressure_stress_level"]
        .fillna("Unclassified")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    distribution.columns = ["Pressure Level", "Days"]
    colors = [pressure_level_color(level) for level in distribution["Pressure Level"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=distribution["Pressure Level"],
                values=distribution["Days"],
                hole=0.58,
                marker={"colors": colors},
                textinfo="label+percent",
                hovertemplate="%{label}<br>Days: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    return apply_plotly_layout(
        fig,
        title="Pressure Stress Level Distribution",
        height=COMPACT_HEIGHT,
    )


def create_primary_driver_distribution(df: pd.DataFrame) -> go.Figure:
    """Create primary pressure driver distribution chart."""
    if df.empty or "primary_pressure_driver" not in df.columns:
        return empty_figure("Primary pressure driver distribution is unavailable because required columns are missing.")

    driver_df = (
        df["primary_pressure_driver"]
        .fillna("Unclassified")
        .astype(str)
        .value_counts()
        .head(8)
        .sort_values()
        .reset_index()
    )
    driver_df.columns = ["Primary Pressure Driver", "Days"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=driver_df["Days"],
                y=driver_df["Primary Pressure Driver"],
                orientation="h",
                marker={"color": COLORS["blue"]},
                hovertemplate="%{y}<br>Days: %{x:,.0f}<extra></extra>",
            )
        ]
    )

    return apply_plotly_layout(
        fig,
        title="Primary Pressure Driver Distribution",
        height=COMPACT_HEIGHT,
        xaxis_title="Days",
    )


# ============================================================
# 13. UI Rendering Helpers
# ============================================================

def render_section_intro(title: str, description: str) -> None:
    """Render consistent module section intro."""
    st.markdown(
        f"""
        <div class="hhs-section-intro">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the main dashboard header."""
    st.markdown(
        "<div class='hhs-kicker-native'>Healthcare Operations Capacity Monitor</div>",
        unsafe_allow_html=True,
    )

    logo_col, title_col = st.columns([1.42, 6.58], gap="medium")

    with logo_col:
        if HHS_LOGO_PATH.exists():
            st.image(str(HHS_LOGO_PATH), width=146)
        else:
            st.markdown("### HHS")

    with title_col:
        st.markdown(
            """
            <h1 class="hhs-main-title-native">
                System Capacity & Care Load Analytics for Unaccompanied Children
            </h1>
            <p class="hhs-main-description-native">
                Monitoring CBP custody, HHS care load, transfer pressure, discharge relief,
                backlog accumulation, and system capacity stress across the UAC care pipeline.
            </p>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_cards(kpis: dict[str, dict[str, Any]]) -> None:
    """Render five guideline KPI cards."""
    kpi_columns = st.columns(5, gap="large")
    kpi_items = list(kpis.values())

    for col, item in zip(kpi_columns, kpi_items):
        with col:
            st.metric(
                label=item["label"],
                value=item["display"],
                delta=item["delta"],
            )
            
def render_guideline_kpi_cards(
    kpis: dict[str, dict[str, Any]],
    granularity: str,
) -> None:
    """Render the five official project-guideline KPI cards with custom enterprise structure."""

    if not kpis:
        st.info("Guideline KPI summary is unavailable for the selected filters.")
        return

    period_label = {
        "Daily": "day",
        "Weekly": "week",
        "Monthly": "month",
    }.get(granularity, "period")

    def get_status_class(status_text: str) -> str:
        """Map KPI status text to a visual status class."""
        normalized_status = str(status_text).strip().lower()

        if any(term in normalized_status for term in ["relief", "stable", "controlled", "current"]):
            return "positive"

        if any(term in normalized_status for term in ["watch", "near balance"]):
            return "watch"

        if any(term in normalized_status for term in ["pressure", "volatile", "accumulating", "gap"]):
            return "pressure"

        return "neutral"

    def format_delta_text(delta_text: Any) -> str:
        """Clean KPI delta/context text for display."""
        clean_delta = str(delta_text).strip()

        if not clean_delta or clean_delta.lower() in {"nan", "none"}:
            return "No prior comparison available"

        return clean_delta

    def render_single_kpi_card(kpi_key: str, card_rank: str) -> None:
        """Render one KPI card using controlled class names."""
        item = kpis.get(kpi_key)

        if not item:
            return

        label = item.get("label", "KPI")
        display_value = item.get("display", "0")
        raw_value = item.get("value", 0)
        delta = format_delta_text(item.get("delta", ""))
        status = item.get("status", "Status unavailable")
        meaning = item.get("meaning", "Operational meaning unavailable.")
        status_class = get_status_class(status)

        if kpi_key == "net_intake_pressure":
            display_value = f"{format_compact_number(raw_value, 1)} / {period_label}"

        card_html = (
            f'<div class="hhs-guideline-kpi-card hhs-guideline-kpi-card--{status_class} hhs-guideline-kpi-card--{card_rank}">'
            f'<div class="hhs-guideline-kpi-card-top">'
            f'<div class="hhs-guideline-kpi-label">{label}</div>'
            f'<div class="hhs-guideline-kpi-status hhs-guideline-kpi-status--{status_class}">{status}</div>'
            f'</div>'
            f'<div class="hhs-guideline-kpi-value">{display_value}</div>'
            f'<div class="hhs-guideline-kpi-delta">{delta}</div>'
            f'<div class="hhs-guideline-kpi-meaning">{meaning}</div>'
            f'</div>'
        )

        st.markdown(card_html, unsafe_allow_html=True)

    intro_html = (
        '<div class="hhs-guideline-kpi-section-intro">'
        'Official capacity-monitoring KPIs based on the selected date range and timeline granularity.'
        '</div>'
    )

    st.markdown(intro_html, unsafe_allow_html=True)

    top_row = st.columns(3, gap="large")

    with top_row[0]:
        render_single_kpi_card("total_children_under_care", "primary")

    with top_row[1]:
        render_single_kpi_card("net_intake_pressure", "primary")

    with top_row[2]:
        render_single_kpi_card("discharge_offset_ratio", "primary")

    bottom_row = st.columns(2, gap="large")

    with bottom_row[0]:
        render_single_kpi_card("care_load_volatility_index", "secondary")

    with bottom_row[1]:
        render_single_kpi_card("backlog_accumulation_rate", "secondary")
            
def calculate_selected_metric_kpis(
    df: pd.DataFrame,
    selected_metrics: list[str],
) -> list[dict[str, Any]]:
    """Calculate dynamic KPI cards from selected sidebar metric toggles."""
    metric_config = {
        "Total System Load": {
            "column": "total_system_load",
            "aggregation": "latest",
            "suffix": "",
            "description": "Latest system-wide care load",
        },
        "CBP Custody": {
            "column": "cbp_custody",
            "aggregation": "latest",
            "suffix": "",
            "description": "Latest active CBP custody load",
        },
        "HHS Care": {
            "column": "hhs_care",
            "aggregation": "latest",
            "suffix": "",
            "description": "Latest active HHS care load",
        },
        "Transfers into HHS": {
            "column": "transfers_to_hhs",
            "aggregation": "sum",
            "suffix": "",
            "description": "Total transfers into HHS",
        },
        "Discharges from HHS": {
            "column": "hhs_discharged",
            "aggregation": "sum",
            "suffix": "",
            "description": "Total HHS discharges",
        },
        "Net Daily Intake": {
            "column": "net_daily_intake",
            "aggregation": "mean",
            "suffix": " / period",
            "description": "Average intake pressure",
        },
        "Care Load Growth Rate": {
            "column": "care_load_growth_rate_pct",
            "aggregation": "mean",
            "suffix": "%",
            "description": "Average care load growth",
        },
        "Backlog Severity Score": {
            "column": "backlog_severity_score",
            "aggregation": "mean",
            "suffix": "",
            "description": "Average backlog severity",
        },
        "Final Pressure Stress Score": {
            "column": "final_pressure_stress_score",
            "aggregation": "mean",
            "suffix": "",
            "description": "Average pressure stress score",
        },
    }

    cards = []

    if df.empty:
        return cards

    for metric_label in selected_metrics:
        config = metric_config.get(metric_label)

        if not config:
            continue

        column = config["column"]

        if column not in df.columns:
            continue

        clean_series = pd.to_numeric(df[column], errors="coerce").dropna()

        if clean_series.empty:
            value = 0.0
            previous = 0.0
        else:
            if config["aggregation"] == "latest":
                value = safe_float(clean_series.iloc[-1])
                previous = safe_float(clean_series.iloc[-2]) if len(clean_series) >= 2 else value

            elif config["aggregation"] == "sum":
                value = safe_float(clean_series.sum())
                half_point = max(len(clean_series) // 2, 1)
                previous = safe_float(clean_series.iloc[:half_point].sum())

            elif config["aggregation"] == "mean":
                value = safe_float(clean_series.mean())
                half_point = max(len(clean_series) // 2, 1)
                previous = safe_float(clean_series.iloc[:half_point].mean())

            else:
                value = safe_float(clean_series.iloc[-1])
                previous = value

        delta = value - previous

        if config["suffix"] == "%":
            display_value = format_pct(value, 1)
        elif abs(value) < 10 and config["aggregation"] == "mean":
            display_value = f"{value:.2f}{config['suffix']}"
        else:
            display_value = f"{format_compact_number(value, 1)}{config['suffix']}"

        cards.append(
            {
                "label": metric_label,
                "value": display_value,
                "delta": f"{delta:+,.1f} vs baseline",
                "description": config["description"],
            }
        )

    return cards

def render_selected_metric_kpi_cards(metric_cards: list[dict[str, Any]]) -> None:
    """Render KPI cards based on selected sidebar metrics."""
    if not metric_cards:
        st.info("Select at least one metric from the sidebar to view dynamic KPI cards.")
        return

    visible_cards = metric_cards[:5]
    kpi_columns = st.columns(len(visible_cards), gap="large")

    for col, card in zip(kpi_columns, visible_cards):
        with col:
            st.metric(
                label=card["label"],
                value=card["value"],
                delta=card["delta"],
            )

    if len(metric_cards) > 5:
        st.caption("Showing first 5 selected metrics as KPI cards. Remaining selected metrics are available in the trend chart.")            


def render_load_summary_cards(df: pd.DataFrame) -> None:
    """Render dynamic system load summary cards."""
    if df.empty or "total_system_load" not in df.columns:
        st.info("System load summary is unavailable because total system load data is missing.")
        return

    latest_load = latest_value(df, "total_system_load")
    avg_load = safe_float(df["total_system_load"].mean())
    peak_load = safe_float(df["total_system_load"].max())
    min_load = safe_float(df["total_system_load"].min())

    peak_date = (
        df.loc[df["total_system_load"].idxmax(), "date"].strftime("%d %b %Y")
        if not df["total_system_load"].dropna().empty
        else "N/A"
    )

    col_1, col_2, col_3, col_4, col_5 = st.columns(5, gap="medium")

    with col_1:
        st.metric("Latest Load", format_compact_number(latest_load))
    with col_2:
        st.metric("Average Load", format_compact_number(avg_load))
    with col_3:
        st.metric("Peak Load", format_compact_number(peak_load))
    with col_4:
        st.metric("Lowest Load", format_compact_number(min_load))
    with col_5:
        st.metric("Peak Load Date", peak_date)


def build_filtered_records_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a curated analytical records table."""
    if df.empty:
        return pd.DataFrame()

    column_labels = {
        "date": "Date",
        "total_system_load": "Total System Load",
        "cbp_custody": "CBP Custody",
        "hhs_care": "HHS Care",
        "transfers_to_hhs": "Transfers into HHS",
        "hhs_discharged": "Discharges from HHS",
        "net_daily_intake": "Net Daily Intake",
        "care_load_growth_rate_pct": "Care Load Growth Rate %",
        "backlog_severity_score": "Backlog Severity",
        "final_pressure_stress_level": "Pressure Stress Level",
        "primary_pressure_driver": "Primary Pressure Driver",
        "operational_response_category": "Operational Response Category",
    }

    available_columns = [column for column in column_labels if column in df.columns]

    table = df[available_columns].copy()
    table = table.rename(columns=column_labels)

    if "Date" in table.columns:
        table["Date"] = pd.to_datetime(table["Date"], errors="coerce").dt.strftime("%d %b %Y")

    numeric_columns = table.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if "%" in column:
            table[column] = table[column].map(lambda x: f"{safe_float(x):.2f}%")
        elif "Rate" in column or "Severity" in column:
            table[column] = table[column].map(lambda x: f"{safe_float(x):.2f}")
        else:
            table[column] = table[column].map(lambda x: f"{safe_float(x):,.0f}")

    return table


# ============================================================
# 14. Sidebar Controls
# ============================================================

def render_sidebar(df: pd.DataFrame) -> tuple[tuple[Any, Any] | None, str, list[str]]:
    """
    Render the final compact enterprise healthcare analytics sidebar.

    Approved sidebar capabilities:
    1. Date Range Selector
    2. Time Granularity Filter
    3. Metric Toggles

    The sidebar is intentionally Streamlit-native for maximum layout stability.
    """

    with st.sidebar:
        selected_date_range = None

        # ----------------------------------------------------
        # Compact Brand Area
        # ----------------------------------------------------
        if UNIFIED_MENTOR_LOGO_PATH.exists():
            st.image(str(UNIFIED_MENTOR_LOGO_PATH), width=188)

        st.markdown("### Healthcare Capacity Analytics")
        st.caption("CBP–HHS care load monitoring")

        st.divider()

        # ----------------------------------------------------
        # Dashboard Filters
        # ----------------------------------------------------
        st.markdown("### Dashboard Filters")
        st.caption("Date range, timeline level, and active metrics.")

        if df.empty or "date" not in df.columns:
            st.info("Date filter unavailable because no valid date column was found.")
        else:
            valid_dates = df["date"].dropna()

            if valid_dates.empty:
                st.info("Date filter unavailable because no valid reporting dates were found.")
            else:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                selected_date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_selector",
                )

        selected_granularity = st.selectbox(
            "Time Granularity",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            key="time_granularity",
        )

        metric_options = {
            "Total System Load": "total_system_load",
            "CBP Custody": "cbp_custody",
            "HHS Care": "hhs_care",
            "Transfers into HHS": "transfers_to_hhs",
            "Discharges from HHS": "hhs_discharged",
            "Net Daily Intake": "net_daily_intake",
            "Care Load Growth Rate": "care_load_growth_rate_pct",
            "Backlog Severity Score": "backlog_severity_score",
            "Final Pressure Stress Score": "final_pressure_stress_score",
        }

        available_metric_labels = [
            label
            for label, column in metric_options.items()
            if column in df.columns
        ]

        default_metrics = [
            label
            for label in [
                "Total System Load",
                "CBP Custody",
                "HHS Care",
                "Net Daily Intake",
            ]
            if label in available_metric_labels
        ]

        selected_metrics = st.multiselect(
            "Metric Toggles",
            options=available_metric_labels,
            default=default_metrics,
            key="metric_toggles",
            help="Selected metrics update the dynamic KPI cards and selected metric trend chart.",
        )

        st.divider()

        # ----------------------------------------------------
        # Compact Scope Summary
        # ----------------------------------------------------
        st.markdown("### Monitoring Scope")

        st.caption(
            "Care load, flow balance, backlog pressure, and system capacity stress."
        )

        st.markdown(
            """
            <div class="sidebar-scope-mini">
                <div><strong>Care Load</strong><span>CBP + HHS responsibility</span></div>
                <div><strong>Flow Balance</strong><span>Transfers vs discharges</span></div>
                <div><strong>Backlog</strong><span>Positive net intake pressure</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_date_range, selected_granularity, selected_metrics


# ============================================================
# 15. Metric Toggle Chart
# ============================================================

def create_selected_metric_trend(df: pd.DataFrame, selected_metrics: list[str]) -> go.Figure:
    """Create dynamic selected metric trend based on sidebar toggles."""
    metric_map = {
        "Total System Load": ("total_system_load", COLORS["navy"]),
        "CBP Custody": ("cbp_custody", COLORS["slate"]),
        "HHS Care": ("hhs_care", COLORS["blue"]),
        "Transfers into HHS": ("transfers_to_hhs", COLORS["amber"]),
        "Discharges from HHS": ("hhs_discharged", COLORS["teal"]),
        "Net Daily Intake": ("net_daily_intake", COLORS["orange"]),
        "Care Load Growth Rate": ("care_load_growth_rate_pct", COLORS["purple"]),
        "Backlog Severity Score": ("backlog_severity_score", COLORS["red"]),
        "Final Pressure Stress Score": ("final_pressure_stress_score", COLORS["green"]),
    }

    if df.empty or not selected_metrics:
        return empty_figure("Select at least one metric from the sidebar to view dynamic metric trends.")

    fig = go.Figure()

    added_any_trace = False

    for label in selected_metrics:
        column, color = metric_map.get(label, (None, COLORS["slate"]))

        if column and column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[column],
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 2.5},
                    hovertemplate=f"%{{x|%b %d, %Y}}<br>{label}: %{{y:,.2f}}<extra></extra>",
                )
            )
            added_any_trace = True

    if not added_any_trace:
        return empty_figure("Selected metrics are unavailable in the processed dataset.")

    return apply_plotly_layout(
        fig,
        title="Selected Metric Trend Preview",
        yaxis_title="Selected Metric Value",
    )
    
# ============================================================
# Module 1 — Stable Hybrid Rendering Layer
# Phase 1: Renderer Stabilization
# ============================================================

def hhs_escape_html(value: Any) -> str:
    """Escape dynamic values before inserting them into controlled HTML cards."""
    text = str(value if value is not None else "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def hhs_compact_html(html: str) -> str:
    """Convert HTML into one safe inline string for Streamlit rendering."""
    import re
    import textwrap

    clean_html = " ".join(
        line.strip()
        for line in textwrap.dedent(str(html)).splitlines()
        if line.strip()
    )
    clean_html = re.sub(r">\s+<", "><", clean_html)
    return clean_html.strip()


def hhs_render_html(html: str) -> None:
    """Render compact controlled HTML for Module 1 without Markdown code-block leakage."""
    clean_html = hhs_compact_html(html)

    if not clean_html:
        return

    if hasattr(st, "html"):
        st.html(clean_html)
        return

    st.markdown(clean_html, unsafe_allow_html=True)


def get_module1_severity_class(intelligence: dict[str, Any] | None) -> str:
    """Return a safe CSS severity class for Module 1 components."""
    if not intelligence:
        return "neutral"

    status = intelligence.get("status", {})
    severity = str(status.get("severity", "neutral")).strip().lower()

    allowed = {
        "critical",
        "pressure",
        "elevated",
        "watch",
        "relief",
        "stable",
        "neutral",
    }

    return severity if severity in allowed else "neutral"


def hhs_module1_safe_class(value: Any, default: str = "neutral") -> str:
    """Create a safe CSS class token from trusted internal values."""
    import re

    token = str(value if value is not None else default).strip().lower()
    token = re.sub(r"[^a-z0-9_-]+", "-", token).strip("-")
    return token or default


def format_module1_signed_delta(value: Any, decimals: int = 0) -> str:
    """Format signed load movement values for executive cards."""
    number = safe_float(value)

    if decimals:
        return f"{number:+,.{decimals}f}"

    return f"{number:+,.0f}"


def format_module1_direction(value: Any) -> str:
    """Return a human-readable movement label."""
    number = safe_float(value)

    if number > 0:
        return "Increase"

    if number < 0:
        return "Decrease"

    return "No Change"



def render_module1_command_header(intelligence: dict[str, Any]) -> None:
    """Render the dynamic Module 1 executive command header."""
    status = intelligence.get("status", {})
    metrics = intelligence.get("metrics", {})
    insights = intelligence.get("insights", {})
    context = intelligence.get("granularity_context", {})

    severity = get_module1_severity_class(intelligence)

    status_label = hhs_escape_html(status.get("status", "Stable"))
    summary = hhs_escape_html(
        status.get(
            "summary",
            "System load condition has been calculated from the selected filter state.",
        )
    )
    leadership_focus = hhs_escape_html(
        status.get(
            "leadership_focus",
            "Maintain monitoring and review operational movement.",
        )
    )
    latest_date = hhs_escape_html(metrics.get("latest_date_display", "Unavailable"))
    period_label = hhs_escape_html(str(context.get("period_label", "daily")).title())
    status_reasoning = hhs_escape_html(
        insights.get(
            "status_reasoning",
            insights.get(
                "executive_summary",
                "Operational status reflects the current selected filter state.",
            ),
        )
    )

    html = f"""
    <section class="hhs-m1-command-shell hhs-m1-status-{severity}">
        <div class="hhs-m1-command-header">
            <div class="hhs-m1-command-eyebrow">Module 1 · Fully Dynamic Healthcare Operations Analytics</div>
            <div class="hhs-m1-command-topline">
                <div class="hhs-m1-command-title-block">
                    <h2 class="hhs-m1-command-title">System Load Overview</h2>
                    <p class="hhs-m1-command-subtitle">Filter-aware care-load intelligence for CBP custody, HHS care, intake pressure, discharge relief, capacity range, risk stability, and executive action.</p>
                </div>
                <div class="hhs-m1-status-badge hhs-m1-status-badge-{severity}">{status_label}</div>
            </div>
            <div class="hhs-m1-command-summary">
                <div class="hhs-m1-command-summary-main">{summary}</div>
                <div class="hhs-m1-command-summary-support">{status_reasoning}</div>
            </div>
            <div class="hhs-m1-command-focus-row">
                <div class="hhs-m1-focus-item"><span class="hhs-m1-focus-label">Leadership Focus</span><strong>{leadership_focus}</strong></div>
                <div class="hhs-m1-focus-item"><span class="hhs-m1-focus-label">Latest Period</span><strong>{latest_date}</strong></div>
                <div class="hhs-m1-focus-item"><span class="hhs-m1-focus-label">Current View</span><strong>{period_label}</strong></div>
            </div>
        </div>
    </section>
    """
    hhs_render_html(html)

def render_module1_empty_command_state(intelligence: dict[str, Any]) -> None:
    """Render empty state when Module 1 has no usable data."""
    status = intelligence.get("status", {})
    recommendations = intelligence.get("recommendations", {})

    empty_summary = hhs_escape_html(
        status.get(
            "summary",
            "No usable data is available for the selected filter state.",
        )
    )
    empty_action = hhs_escape_html(
        recommendations.get(
            "leadership",
            "Validate the selected date range and dataset availability.",
        )
    )

    html = f"""
    <div class="hhs-m1-empty-state">
        <div class="hhs-m1-empty-title">Module 1 Intelligence Unavailable</div>
        <p>{empty_summary}</p>
        <strong>{empty_action}</strong>
    </div>
    """
    hhs_render_html(html)


def render_module1_primary_kpi_strip(intelligence: dict[str, Any]) -> None:
    """Render primary Module 1 command-center KPI strip."""
    metrics = intelligence.get("metrics", {})
    status = intelligence.get("status", {})
    context = intelligence.get("granularity_context", {})

    severity = get_module1_severity_class(intelligence)
    previous_label = context.get("previous_label", "vs prior period")
    average_label = context.get("average_label", "average per period")

    current_load = safe_float(metrics.get("current_load"))
    load_delta = safe_float(metrics.get("load_delta"))
    load_delta_pct = safe_float(metrics.get("load_delta_pct"))
    load_position_pct = safe_float(metrics.get("load_position_pct"))
    net_intake_pressure = safe_float(metrics.get("net_intake_pressure"))
    volatility_index = safe_float(metrics.get("volatility_index"))
    backlog_pressure_share = safe_float(metrics.get("backlog_pressure_share"))
    discharge_offset_ratio = safe_float(metrics.get("discharge_offset_ratio"))

    rolling_trend_label = str(metrics.get("rolling_trend_label", "Stable Trend"))
    rolling_trend_direction = str(metrics.get("rolling_trend_direction", "Stable"))
    status_name = str(status.get("status", "Stable"))

    cards = [
        {
            "label": "Current System Load",
            "value": format_compact_number(current_load),
            "meta": "Total responsibility across CBP custody and HHS care",
            "signal": status_name,
            "class_name": "primary",
        },
        {
            "label": "Load Change",
            "value": format_module1_signed_delta(load_delta),
            "meta": f"{format_pct(load_delta_pct, 1)} · {previous_label}",
            "signal": format_module1_direction(load_delta),
            "class_name": "movement",
        },
        {
            "label": "Load Position vs Peak",
            "value": format_pct(load_position_pct, 1),
            "meta": "Current load as share of selected-period peak",
            "signal": "Near Peak" if load_position_pct >= 90 else "Range Context",
            "class_name": "capacity",
        },
        {
            "label": "Pressure / Relief State",
            "value": format_compact_number(net_intake_pressure, 1),
            "meta": average_label,
            "signal": "Pressure" if net_intake_pressure > 0 else "Relief" if net_intake_pressure < 0 else "Balanced",
            "class_name": "pressure",
        },
        {
            "label": "Rolling Trend",
            "value": rolling_trend_label,
            "meta": "Short-term vs longer trend direction",
            "signal": rolling_trend_direction,
            "class_name": "trend",
        },
        {
            "label": "Risk Stability",
            "value": format_pct(volatility_index, 1),
            "meta": f"Backlog {format_pct(backlog_pressure_share, 1)} · Offset {format_ratio(discharge_offset_ratio)}",
            "signal": "Volatile" if volatility_index > 5 else "Watch" if volatility_index > 2 else "Controlled",
            "class_name": "risk",
        },
    ]

    card_html = []
    for card in cards:
        class_name = hhs_module1_safe_class(card.get("class_name", "metric"), "metric")
        card_html.append(
            f"""
            <article class="hhs-m1-kpi-card hhs-m1-kpi-card-{class_name} hhs-m1-status-{severity}">
                <div class="hhs-m1-kpi-top"><span>{hhs_escape_html(card['label'])}</span><em>{hhs_escape_html(card['signal'])}</em></div>
                <div class="hhs-m1-kpi-value">{hhs_escape_html(card['value'])}</div>
                <div class="hhs-m1-kpi-meta">{hhs_escape_html(card['meta'])}</div>
            </article>
            """
        )

    hhs_render_html(f"<div class='hhs-m1-kpi-strip'>{''.join(card_html)}</div>")



def render_module1_operational_evidence_bar(intelligence: dict[str, Any]) -> None:
    """Render a dynamic evidence bar explaining the operating status with filter-aware signals."""
    metrics = intelligence.get("metrics", {})
    context = intelligence.get("granularity_context", {})
    severity = get_module1_severity_class(intelligence)

    dominant_risk = metrics.get("dominant_risk_signal", {}) or {}
    selected_diag = metrics.get("selected_metric_diagnostics", {}) or {}

    evidence_cards = [
        {
            "label": "Selected Window",
            "value": f"{int(safe_float(metrics.get('selected_periods')))} {context.get('period_name', 'period')}(s)",
            "meta": f"Latest: {metrics.get('latest_date_display', 'Unavailable')}",
        },
        {
            "label": "Peak Context",
            "value": format_compact_number(metrics.get("peak_load")),
            "meta": f"Peak date: {metrics.get('peak_date_display', 'Unavailable')}",
        },
        {
            "label": "Flow Offset",
            "value": format_ratio(metrics.get("discharge_offset_ratio")),
            "meta": f"Net intake avg: {format_compact_number(metrics.get('net_intake_pressure'), 1)}",
        },
        {
            "label": "Dominant Risk",
            "value": dominant_risk.get("label", "No Dominant Risk"),
            "meta": dominant_risk.get("value", "Routine monitoring"),
        },
        {
            "label": "Diagnostic Metrics",
            "value": f"{int(safe_float(selected_diag.get('available_count')))} active",
            "meta": f"{selected_diag.get('alignment_state', 'Unavailable')} signals",
        },
    ]

    cards_html = "".join(
        f"""
        <article class="hhs-m1-evidence-card">
            <span>{hhs_escape_html(card['label'])}</span>
            <strong>{hhs_escape_html(card['value'])}</strong>
            <em>{hhs_escape_html(card['meta'])}</em>
        </article>
        """
        for card in evidence_cards
    )

    html = f"""
    <section class="hhs-m1-evidence-grid hhs-m1-status-{severity}">
        {cards_html}
    </section>
    """

    hhs_render_html(html)

def render_module1_section_anchor(kicker: str, title: str) -> None:
    """Render a clean section anchor for Module 1 command-center sections."""
    safe_kicker = hhs_escape_html(kicker)
    safe_title = hhs_escape_html(title)

    html = f"""
    <div class="hhs-m1-section-anchor">
        <span>{safe_kicker}</span>
        <strong>{safe_title}</strong>
    </div>
    """
    hhs_render_html(html)


def render_module1_insight_recommendation_block(
    title: str,
    insight: str,
    recommendation: str,
    severity: str,
    insight_label: str = "Dynamic Insight",
    recommendation_label: str = "Recommended Action",
) -> None:
    """Render dynamic insight and recommendation block below a Module 1 visualization."""
    safe_severity = hhs_module1_safe_class(severity, "neutral")

    html = f"""
    <section class="hhs-m1-intel-block hhs-m1-status-{safe_severity}">
        <div class="hhs-m1-intel-title">{hhs_escape_html(title)}</div>
        <div class="hhs-m1-intel-grid">
            <article class="hhs-m1-insight-card"><span>{hhs_escape_html(insight_label)}</span><p>{hhs_escape_html(insight)}</p></article>
            <article class="hhs-m1-reco-card"><span>{hhs_escape_html(recommendation_label)}</span><p>{hhs_escape_html(recommendation)}</p></article>
        </div>
    </section>
    """
    hhs_render_html(html)


def render_module1_visual_explanation(
    title: str,
    what_it_shows: str,
    how_to_read: str,
    focus: str,
    severity: str,
) -> None:
    """Render dynamic explanation layer for a Module 1 visual."""
    safe_severity = hhs_module1_safe_class(severity, "neutral")

    html = f"""
    <section class="hhs-m1-explanation hhs-m1-status-{safe_severity}">
        <div class="hhs-m1-explanation-title">{hhs_escape_html(title)}</div>
        <div class="hhs-m1-explanation-grid">
            <article><span>What this shows</span><p>{hhs_escape_html(what_it_shows)}</p></article>
            <article><span>How to read it</span><p>{hhs_escape_html(how_to_read)}</p></article>
            <article><span>Decision focus</span><p>{hhs_escape_html(focus)}</p></article>
        </div>
    </section>
    """
    hhs_render_html(html)


def render_module1_decision_support_closeout(intelligence: dict[str, Any]) -> None:
    """Render final executive decision-support panel for Module 1."""
    recommendations = intelligence.get("recommendations", {})
    insights = intelligence.get("insights", {})
    status = intelligence.get("status", {})

    severity = get_module1_severity_class(intelligence)
    status_label = hhs_escape_html(status.get("status", "Stable"))
    executive_summary = hhs_escape_html(
        insights.get(
            "executive_summary",
            status.get(
                "summary",
                "Operational status has been calculated from the selected period.",
            ),
        )
    )

    decision_items = [
        ("Leadership Priority", recommendations.get("leadership", "Maintain leadership monitoring.")),
        ("Operations Monitoring", recommendations.get("operations", "Continue operational monitoring.")),
        ("Planning Action", recommendations.get("planning", "Maintain planning readiness.")),
        ("Risk Watch", recommendations.get("risk_watch", "Continue standard risk monitoring.")),
    ]

    item_html = "".join(
        f"<article><span>{hhs_escape_html(label)}</span><p>{hhs_escape_html(text)}</p></article>"
        for label, text in decision_items
    )
    period_note = hhs_escape_html(
        recommendations.get(
            "period_note",
            "Recommendations reflect the current selected filter state.",
        )
    )

    html = f"""
    <section class="hhs-m1-decision-panel hhs-m1-status-{severity}">
        <div class="hhs-m1-decision-header"><span>Executive Decision Support</span><strong>{status_label} Operating View</strong></div>
        <div class="hhs-m1-decision-summary">{executive_summary}</div>
        <div class="hhs-m1-decision-grid">{item_html}</div>
        <div class="hhs-m1-period-note">{period_note}</div>
    </section>
    """
    hhs_render_html(html)



def render_module_1_command_center(
    df_visual: pd.DataFrame,
    df_filtered_daily: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> None:
    """Render the complete production-grade Module 1 healthcare operations command center."""
    selected_metrics = selected_metrics or []

    intelligence = calculate_module1_intelligence(
        df_visual=df_visual,
        df_filtered_daily=df_filtered_daily,
        selected_granularity=selected_granularity,
        selected_metrics=selected_metrics,
    )

    if not intelligence.get("has_data", False):
        render_module1_empty_command_state(intelligence)
        return

    severity = get_module1_severity_class(intelligence)
    insights = intelligence.get("insights", {})
    recommendations = intelligence.get("recommendations", {})
    context = intelligence.get("granularity_context", {})
    period_label = context.get("period_label", "daily")

    render_module1_command_header(intelligence)
    render_module1_primary_kpi_strip(intelligence)
    render_module1_operational_evidence_bar(intelligence)

    render_module1_section_anchor(
        kicker="Trend Intelligence",
        title="Total System Load Movement",
    )
    system_load_fig = create_system_load_trend(
        df=df_visual,
        intelligence=intelligence,
        selected_granularity=selected_granularity,
    )
    st.plotly_chart(system_load_fig, use_container_width=True, config=PLOTLY_CONFIG)

    render_module1_insight_recommendation_block(
        title="System Load Trend Intelligence",
        insight=insights.get("trend", "Trend insight is unavailable for the selected period."),
        recommendation=recommendations.get("leadership", "Maintain leadership monitoring."),
        severity=severity,
        insight_label="Trend Insight",
        recommendation_label="Leadership Recommendation",
    )
    render_module1_visual_explanation(
        title="How to Read the System Load Trend",
        what_it_shows="This chart tracks total system load across CBP custody and HHS care for the selected date range and selected time granularity.",
        how_to_read="Use the main line for total load direction, rolling averages for trend confirmation, and peak/latest markers for current range position.",
        focus=f"Focus on whether the latest {period_label} value is above average, near peak, stable, or moving toward relief.",
        severity=severity,
    )

    render_module1_section_anchor(
        kicker="Movement Intelligence",
        title="Pressure vs Relief Analysis",
    )
    movement_fig = create_growth_rate_trend(
        df=df_visual,
        intelligence=intelligence,
        selected_granularity=selected_granularity,
    )
    st.plotly_chart(movement_fig, use_container_width=True, config=PLOTLY_CONFIG)

    render_module1_insight_recommendation_block(
        title="Pressure vs Relief Movement Intelligence",
        insight=insights.get("pressure", "Pressure insight is unavailable for the selected period."),
        recommendation=recommendations.get("operations", "Continue operational monitoring."),
        severity=severity,
        insight_label="Pressure Insight",
        recommendation_label="Operations Recommendation",
    )
    render_module1_visual_explanation(
        title="How to Read Pressure vs Relief Movement",
        what_it_shows="This chart separates positive care-load growth from relief movement, showing whether the system burden is increasing or easing.",
        how_to_read="Positive movement indicates pressure buildup, negative movement indicates relief, and near-zero movement indicates balanced movement.",
        focus="Focus on whether pressure periods are isolated or sustained across the selected operational window.",
        severity=severity,
    )

    render_module1_section_anchor(
        kicker="Capacity and Risk Intelligence",
        title="Range Position and Stability Signals",
    )
    capacity_col, risk_col = st.columns([1, 1], gap="large")
    with capacity_col:
        capacity_fig = create_capacity_position_view(intelligence)
        st.plotly_chart(capacity_fig, use_container_width=True, config=PLOTLY_CONFIG)
    with risk_col:
        risk_fig = create_risk_stability_signal(intelligence)
        st.plotly_chart(risk_fig, use_container_width=True, config=PLOTLY_CONFIG)

    capacity_risk_insight = (
        f"{insights.get('capacity', 'Capacity insight is unavailable.')} "
        f"{insights.get('risk', 'Risk insight is unavailable.')}"
    )
    render_module1_insight_recommendation_block(
        title="Capacity and Risk Intelligence",
        insight=capacity_risk_insight,
        recommendation=recommendations.get("planning", "Maintain planning readiness."),
        severity=severity,
        insight_label="Capacity + Risk Insight",
        recommendation_label="Planning Recommendation",
    )
    render_module1_visual_explanation(
        title="How to Read Capacity and Risk Signals",
        what_it_shows="The capacity view compares latest load against low, average, and peak values. The risk view summarizes volatility, backlog pressure, offset gap, and positive intake streak.",
        how_to_read="Use capacity position to understand range pressure, and use risk signals to detect instability that may be hidden behind averages.",
        focus="Focus on near-peak load, high backlog pressure, high volatility, or weak discharge offset because these affect operational planning reliability.",
        severity=severity,
    )

    render_module1_decision_support_closeout(intelligence)

    if selected_metrics:
        render_module1_section_anchor(
            kicker="Diagnostic Layer",
            title="Selected Metric Explorer",
        )
        explorer_fig = create_module1_selected_metric_explorer(
            df=df_visual,
            selected_metrics=selected_metrics,
            intelligence=intelligence,
        )
        st.plotly_chart(explorer_fig, use_container_width=True, config=PLOTLY_CONFIG)
        render_module1_insight_recommendation_block(
            title="Selected Metric Diagnostic Intelligence",
            insight=insights.get(
                "selected_metric",
                "Selected metric insight is unavailable for the current filter state.",
            ),
            recommendation=recommendations.get(
                "diagnostic",
                "Use the selected metric explorer as a supporting diagnostic layer.",
            ),
            severity=severity,
            insight_label="Selected Metric Insight",
            recommendation_label="Explorer Guidance",
        )
        render_module1_visual_explanation(
            title="How to Read the Selected Metric Explorer",
            what_it_shows="This supporting view compares only the metrics selected in the sidebar metric toggle.",
            how_to_read="Use it to inspect whether selected operational signals are moving together, diverging, increasing, or easing.",
            focus="Focus on the strongest selected-metric movement, but keep the command-center status anchored to system-load and care-pipeline signals.",
            severity=severity,
        )



# ============================================================
# Module 2–4 — Production Command Center Intelligence Layer
# Safe Mode: Scoped upgrade only for modules 2, 3, and 4
# ============================================================

def hhs_safe_class_token(value: Any, default: str = "neutral") -> str:
    """Return a safe CSS class token for trusted dashboard status values."""
    import re

    token = str(value if value is not None else default).strip().lower()
    token = re.sub(r"[^a-z0-9_-]+", "-", token).strip("-")
    return token or default


def format_signed_number(value: Any, decimals: int = 0) -> str:
    """Format signed values for evidence cards."""
    number = safe_float(value)
    if decimals:
        return f"{number:+,.{decimals}f}"
    return f"{number:+,.0f}"


def format_pp(value: Any, decimals: int = 1) -> str:
    """Format percentage-point values."""
    return f"{safe_float(value):+.{decimals}f} pp"


def safe_latest_date_display(df: pd.DataFrame) -> str:
    """Return latest date display string for command headers."""
    if df.empty or "date" not in df.columns:
        return "Unavailable"
    latest_date = pd.to_datetime(df.sort_values("date")["date"].dropna().tail(1).iloc[0], errors="coerce")
    if pd.isna(latest_date):
        return "Unavailable"
    return latest_date.strftime("%b %d, %Y")


def longest_condition_streak(series: pd.Series, condition: str = "positive") -> int:
    """Calculate the longest positive, negative, or zero streak in a numeric series."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).tolist()
    longest = 0
    current = 0

    for value in values:
        if condition == "positive":
            active = value > 0
        elif condition == "negative":
            active = value < 0
        else:
            active = value == 0

        if active:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def get_status_palette(severity: str) -> dict[str, str]:
    """Return status colors for module 2–4 charts."""
    severity = hhs_safe_class_token(severity)
    palettes = {
        "critical": {"primary": COLORS["red"], "secondary": COLORS["orange"], "accent": COLORS["amber"]},
        "pressure": {"primary": COLORS["red"], "secondary": COLORS["orange"], "accent": COLORS["amber"]},
        "elevated": {"primary": COLORS["orange"], "secondary": COLORS["amber"], "accent": COLORS["red"]},
        "watch": {"primary": COLORS["amber"], "secondary": COLORS["orange"], "accent": COLORS["blue"]},
        "relief": {"primary": COLORS["teal"], "secondary": COLORS["green"], "accent": COLORS["blue"]},
        "stable": {"primary": COLORS["blue"], "secondary": COLORS["teal"], "accent": COLORS["green"]},
        "neutral": {"primary": COLORS["slate"], "secondary": COLORS["blue"], "accent": COLORS["teal"]},
    }
    return palettes.get(severity, palettes["neutral"])


def apply_command_center_chart_layout(
    fig: go.Figure,
    title: str,
    height: int = STANDARD_HEIGHT,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
    show_legend: bool = True,
    legend_bottom: bool = True,
) -> go.Figure:
    """Apply a safe module 2–4 chart layout with extra spacing and verified Plotly properties."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
            "font": {
                "size": 17,
                "color": COLORS["navy"],
                "family": "Inter, Segoe UI, sans-serif",
            },
        },
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        hovermode="x unified",
        margin={"l": 60, "r": 58, "t": 76, "b": 82 if show_legend and legend_bottom else 58},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16 if legend_bottom else 1.08,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10, "color": COLORS["navy"]},
        },
        showlegend=show_legend,
        font={"family": "Inter, Segoe UI, sans-serif", "color": COLORS["navy"]},
    )

    fig.update_xaxes(
        title_text=xaxis_title,
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(148, 163, 184, 0.55)",
        tickfont={"size": 11, "color": COLORS["slate"]},
        rangeslider={"visible": False},
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(100, 116, 139, 0.35)",
        tickfont={"size": 11, "color": COLORS["slate"]},
        title_font={"size": 12, "color": COLORS["slate"]},
    )
    return fig


def calculate_module2_intelligence(
    df_visual: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate CBP vs HHS responsibility and handoff intelligence."""
    selected_metrics = selected_metrics or []
    context = get_granularity_context(selected_granularity)

    if df_visual.empty or not {"date", "cbp_custody", "hhs_care"}.issubset(df_visual.columns):
        return {
            "has_data": False,
            "granularity_context": context,
            "selected_metrics": selected_metrics,
            "status": {
                "status": "Data Limited",
                "severity": "neutral",
                "summary": "CBP/HHS comparison cannot be calculated because required load columns are missing.",
                "leadership_focus": "Validate CBP custody and HHS care fields before using this module.",
            },
            "metrics": {},
            "insights": {},
            "recommendations": {},
        }

    df = df_visual.sort_values("date").copy()
    df["cbp_custody"] = pd.to_numeric(df["cbp_custody"], errors="coerce").fillna(0)
    df["hhs_care"] = pd.to_numeric(df["hhs_care"], errors="coerce").fillna(0)
    df["load_total_for_share"] = df["cbp_custody"] + df["hhs_care"]
    df["hhs_share_pct"] = np.where(df["load_total_for_share"] > 0, (df["hhs_care"] / df["load_total_for_share"]) * 100, 0)
    df["cbp_share_pct"] = np.where(df["load_total_for_share"] > 0, (df["cbp_custody"] / df["load_total_for_share"]) * 100, 0)
    df["load_gap"] = df["hhs_care"] - df["cbp_custody"]

    latest = df.tail(1).iloc[0]
    previous = df.tail(2).head(1).iloc[0] if len(df) >= 2 else latest

    latest_cbp = safe_float(latest["cbp_custody"])
    latest_hhs = safe_float(latest["hhs_care"])
    latest_total = latest_cbp + latest_hhs
    current_hhs_share = safe_float(latest["hhs_share_pct"])
    current_cbp_share = safe_float(latest["cbp_share_pct"])
    average_hhs_share = safe_float(df["hhs_share_pct"].mean())
    average_cbp_share = safe_float(df["cbp_share_pct"].mean())
    hhs_share_delta_pp = current_hhs_share - average_hhs_share
    cbp_share_delta_pp = current_cbp_share - average_cbp_share
    hhs_to_cbp_ratio = latest_hhs / latest_cbp if latest_cbp > 0 else float("inf") if latest_hhs > 0 else 0
    load_gap = latest_hhs - latest_cbp
    previous_gap = safe_float(previous["hhs_care"]) - safe_float(previous["cbp_custody"])
    gap_change = load_gap - previous_gap
    hhs_load_delta = latest_hhs - safe_float(previous["hhs_care"])
    cbp_load_delta = latest_cbp - safe_float(previous["cbp_custody"])

    if current_cbp_share >= 25 or cbp_share_delta_pp >= 5:
        status = {
            "status": "CBP Watch",
            "severity": "watch",
            "summary": "CBP custody is carrying an elevated share of the selected-period care load.",
            "leadership_focus": "Watch intake-side custody pressure and confirm that transfers into HHS remain timely.",
        }
    elif current_hhs_share >= 90 and hhs_load_delta > 0:
        status = {
            "status": "HHS Absorption Pressure",
            "severity": "elevated",
            "summary": "HHS is carrying the dominant care burden and latest HHS load is still increasing.",
            "leadership_focus": "Monitor HHS shelter, staffing, placement, and discharge capacity while load remains concentrated downstream.",
        }
    elif current_hhs_share >= 80:
        status = {
            "status": "HHS-Dominant Care Load",
            "severity": "stable",
            "summary": "The latest responsibility split is concentrated in HHS care, which is expected after CBP-to-HHS transfer movement.",
            "leadership_focus": "Maintain downstream care-capacity monitoring and verify that HHS load does not move toward peak pressure.",
        }
    elif abs(hhs_share_delta_pp) >= 6 or abs(cbp_share_delta_pp) >= 6:
        status = {
            "status": "Transition Pressure",
            "severity": "watch",
            "summary": "The CBP/HHS responsibility mix is shifting compared with the selected-period average.",
            "leadership_focus": "Review whether this shift reflects normal handoff flow or an emerging custody/care bottleneck.",
        }
    else:
        status = {
            "status": "Balanced Handoff",
            "severity": "relief",
            "summary": "CBP custody and HHS care responsibility are moving within a controlled selected-period handoff range.",
            "leadership_focus": "Continue routine handoff monitoring and watch for sudden share movement.",
        }

    if latest_total <= 0:
        status.update({
            "status": "Data Limited",
            "severity": "neutral",
            "summary": "No usable load values are available for the latest selected period.",
            "leadership_focus": "Validate selected date range and source data before using the comparison.",
        })

    selected_relevance = [metric for metric in selected_metrics if metric in {"CBP Custody", "HHS Care", "Total System Load"}]

    if current_hhs_share >= 80:
        situation = (
            f"HHS care holds {format_pct(current_hhs_share, 1)} of the latest total load, "
            f"showing that downstream care facilities carry the main operational responsibility."
        )
    elif current_cbp_share >= 25:
        situation = (
            f"CBP custody holds {format_pct(current_cbp_share, 1)} of the latest total load, "
            f"which requires closer attention to intake-side handoff movement."
        )
    else:
        situation = "The latest CBP/HHS split remains within a controlled handoff pattern for the selected period."

    if hhs_share_delta_pp > 0:
        shift = f"HHS share is {format_pp(hhs_share_delta_pp)} above the selected-period average, indicating a stronger downstream care concentration."
    elif hhs_share_delta_pp < 0:
        shift = f"HHS share is {format_pp(hhs_share_delta_pp)} below the selected-period average, indicating relatively less downstream concentration."
    else:
        shift = "Latest HHS share is aligned with the selected-period average."

    if selected_relevance:
        diagnostic = f"Metric toggles include {', '.join(selected_relevance)}, so the selected diagnostic context directly supports this CBP/HHS comparison."
    else:
        diagnostic = "Metric toggles do not currently prioritize CBP/HHS load metrics; use this module as the primary responsibility-split view."

    recommendations = {
        "leadership": status["leadership_focus"],
        "operations": "Compare CBP custody movement with HHS care movement to detect whether responsibility is shifting upstream or downstream.",
        "planning": "Use HHS share and HHS/CBP ratio to plan downstream shelter, care, staffing, and placement capacity.",
        "risk_watch": "Watch for CBP share increases, rapid HHS share growth, or widening load gaps because these can signal handoff stress.",
        "period_note": f"Module 2 evidence reflects the current {context.get('period_label', 'daily')} view and selected date range.",
    }

    return {
        "has_data": True,
        "granularity_context": context,
        "selected_metrics": selected_metrics,
        "status": status,
        "metrics": {
            "latest_date_display": safe_latest_date_display(df),
            "period_count": len(df),
            "latest_cbp": latest_cbp,
            "latest_hhs": latest_hhs,
            "latest_total": latest_total,
            "current_hhs_share": current_hhs_share,
            "current_cbp_share": current_cbp_share,
            "average_hhs_share": average_hhs_share,
            "average_cbp_share": average_cbp_share,
            "hhs_share_delta_pp": hhs_share_delta_pp,
            "cbp_share_delta_pp": cbp_share_delta_pp,
            "hhs_to_cbp_ratio": hhs_to_cbp_ratio,
            "load_gap": load_gap,
            "gap_change": gap_change,
            "hhs_load_delta": hhs_load_delta,
            "cbp_load_delta": cbp_load_delta,
        },
        "insights": {
            "situation": situation,
            "shift": shift,
            "diagnostic": diagnostic,
            "executive_summary": status["summary"],
        },
        "recommendations": recommendations,
    }


def calculate_module3_intelligence(
    df_visual: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate net intake, discharge relief, and backlog intelligence."""
    selected_metrics = selected_metrics or []
    context = get_granularity_context(selected_granularity)

    required = {"date", "transfers_to_hhs", "hhs_discharged", "net_daily_intake"}
    if df_visual.empty or not required.issubset(df_visual.columns):
        return {
            "has_data": False,
            "granularity_context": context,
            "selected_metrics": selected_metrics,
            "status": {
                "status": "Data Limited",
                "severity": "neutral",
                "summary": "Flow balance cannot be calculated because transfer/discharge fields are missing.",
                "leadership_focus": "Validate flow columns before using intake and backlog intelligence.",
            },
            "metrics": {},
            "insights": {},
            "recommendations": {},
        }

    df = df_visual.sort_values("date").copy()
    for column in ["transfers_to_hhs", "hhs_discharged", "net_daily_intake", "backlog_severity_score"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    total_transfers = safe_float(df["transfers_to_hhs"].sum())
    total_discharges = safe_float(df["hhs_discharged"].sum())
    net_total = total_transfers - total_discharges
    avg_net = safe_float(df["net_daily_intake"].mean())
    latest_net = latest_value(df, "net_daily_intake")
    prior_net = previous_value(df, "net_daily_intake")
    latest_net_delta = latest_net - prior_net
    offset_ratio = total_discharges / total_transfers if total_transfers > 0 else 0.0
    pressure_periods = int((df["net_daily_intake"] > 0).sum())
    relief_periods = int((df["net_daily_intake"] < 0).sum())
    balanced_periods = int((df["net_daily_intake"] == 0).sum())
    total_periods = len(df)
    pressure_share = (pressure_periods / total_periods) * 100 if total_periods else 0.0
    relief_share = (relief_periods / total_periods) * 100 if total_periods else 0.0
    longest_pressure = longest_condition_streak(df["net_daily_intake"], "positive")
    longest_relief = longest_condition_streak(df["net_daily_intake"], "negative")
    latest_backlog = latest_value(df, "backlog_severity_score") if "backlog_severity_score" in df.columns else 0.0
    max_backlog = safe_float(df["backlog_severity_score"].max()) if "backlog_severity_score" in df.columns else 0.0
    prolonged_share = safe_float(df["prolonged_pressure_window_flag"].mean() * 100) if "prolonged_pressure_window_flag" in df.columns else 0.0

    worst_idx = df["net_daily_intake"].idxmax() if not df.empty else None
    best_idx = df["net_daily_intake"].idxmin() if not df.empty else None
    worst_period = df.loc[worst_idx, "date"].strftime("%b %d, %Y") if worst_idx is not None and pd.notna(df.loc[worst_idx, "date"]) else "Unavailable"
    best_period = df.loc[best_idx, "date"].strftime("%b %d, %Y") if best_idx is not None and pd.notna(df.loc[best_idx, "date"]) else "Unavailable"

    if avg_net > 0 and offset_ratio < 0.75 and pressure_share > 60:
        status = {
            "status": "Critical Pressure",
            "severity": "critical",
            "summary": "Transfers are materially exceeding discharges and pressure appears persistent across the selected period.",
            "leadership_focus": "Escalate intake-discharge balance review and prepare near-term capacity relief actions.",
        }
    elif pressure_share > 60 or latest_backlog >= 3:
        status = {
            "status": "Backlog Strain",
            "severity": "pressure",
            "summary": "Positive net intake is frequent or backlog severity is elevated, indicating sustained care-load accumulation.",
            "leadership_focus": "Prioritize discharge throughput, placement movement, and backlog monitoring.",
        }
    elif avg_net > 0 or latest_net > 0:
        status = {
            "status": "Accumulation Risk",
            "severity": "elevated",
            "summary": "Net intake is positive, meaning transfers into HHS are exceeding discharges in the current view.",
            "leadership_focus": "Watch whether pressure persists across coming reporting periods before it becomes backlog strain.",
        }
    elif offset_ratio >= 1 and avg_net < 0:
        status = {
            "status": "Relief-Dominant",
            "severity": "relief",
            "summary": "Discharges are offsetting transfers, producing a relief-oriented flow balance for the selected period.",
            "leadership_focus": "Validate that relief is sustained before relaxing operational monitoring.",
        }
    elif pressure_share > 40 or offset_ratio < 1:
        status = {
            "status": "Pressure Watch",
            "severity": "watch",
            "summary": "Flow balance is near a watch condition because pressure periods or offset weakness are present.",
            "leadership_focus": "Track net intake and discharge offset closely for trend confirmation.",
        }
    else:
        status = {
            "status": "Balanced Flow",
            "severity": "stable",
            "summary": "Transfers and discharges are broadly balanced for the selected period.",
            "leadership_focus": "Continue routine flow monitoring and watch for intake shocks.",
        }

    if avg_net > 0:
        situation = f"Average net intake is positive at {format_compact_number(avg_net, 1)} {context.get('average_label', 'average per period')}, showing accumulation pressure."
    elif avg_net < 0:
        situation = f"Average net intake is negative at {format_compact_number(avg_net, 1)} {context.get('average_label', 'average per period')}, showing discharge relief."
    else:
        situation = "Transfers and discharges are nearly balanced, limiting additional intake pressure."

    if offset_ratio >= 1:
        offset_insight = f"Discharge offset ratio is {format_ratio(offset_ratio)}, meaning discharges are keeping pace with or exceeding transfers."
    elif offset_ratio >= 0.85:
        offset_insight = f"Discharge offset ratio is {format_ratio(offset_ratio)}, near balance but still requiring watch-level monitoring."
    else:
        offset_insight = f"Discharge offset ratio is {format_ratio(offset_ratio)}, indicating a flow relief gap."

    relevant = [metric for metric in selected_metrics if metric in {"Transfers into HHS", "Discharges from HHS", "Net Daily Intake", "Backlog Severity Score"}]
    diagnostic = f"Selected metric focus supports Module 3 because it includes {', '.join(relevant)}." if relevant else "Metric toggles do not currently prioritize flow/backlog metrics; Module 3 still calculates required pressure intelligence from guideline fields."

    recommendations = {
        "leadership": status["leadership_focus"],
        "operations": "Compare transfer volume against discharge volume and inspect whether positive net intake is isolated or sustained.",
        "planning": "Use pressure-period share, longest pressure streak, and offset ratio to plan staffing, placement, and shelter relief buffers.",
        "risk_watch": "Watch for repeated positive net intake, rising backlog severity, and offset ratios below 1.00x.",
        "period_note": f"Module 3 evidence reflects the current {context.get('period_label', 'daily')} view and selected date range.",
    }

    return {
        "has_data": True,
        "granularity_context": context,
        "selected_metrics": selected_metrics,
        "status": status,
        "metrics": {
            "latest_date_display": safe_latest_date_display(df),
            "period_count": total_periods,
            "total_transfers": total_transfers,
            "total_discharges": total_discharges,
            "net_total": net_total,
            "avg_net": avg_net,
            "latest_net": latest_net,
            "latest_net_delta": latest_net_delta,
            "offset_ratio": offset_ratio,
            "pressure_periods": pressure_periods,
            "relief_periods": relief_periods,
            "balanced_periods": balanced_periods,
            "pressure_share": pressure_share,
            "relief_share": relief_share,
            "longest_pressure": longest_pressure,
            "longest_relief": longest_relief,
            "latest_backlog": latest_backlog,
            "max_backlog": max_backlog,
            "prolonged_share": prolonged_share,
            "worst_period": worst_period,
            "best_period": best_period,
        },
        "insights": {
            "situation": situation,
            "offset": offset_insight,
            "diagnostic": diagnostic,
            "executive_summary": status["summary"],
        },
        "recommendations": recommendations,
    }


def kpi_health_score(kpi_key: str, value: float) -> float:
    """Convert each guideline KPI into an executive health score where higher is better."""
    if kpi_key == "net_intake_pressure":
        if value < 0:
            return 92
        if abs(value) <= 5:
            return 82
        if value <= 25:
            return 68
        return 48
    if kpi_key == "care_load_volatility_index":
        if value <= 2:
            return 92
        if value <= 5:
            return 72
        return 48
    if kpi_key == "backlog_accumulation_rate":
        if value <= 40:
            return 90
        if value <= 60:
            return 70
        return 45
    if kpi_key == "discharge_offset_ratio":
        if value >= 1:
            return 92
        if value >= 0.85:
            return 72
        return 45
    if kpi_key == "total_children_under_care":
        return 78 if value > 0 else 0
    return 60


def calculate_module4_intelligence(
    kpis: dict[str, dict[str, Any]],
    df_filtered_daily: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate executive KPI health and pressure-driver intelligence."""
    selected_metrics = selected_metrics or []
    context = get_granularity_context(selected_granularity)

    if not kpis:
        return {
            "has_data": False,
            "granularity_context": context,
            "selected_metrics": selected_metrics,
            "status": {
                "status": "Data Limited",
                "severity": "neutral",
                "summary": "KPI intelligence cannot be calculated because KPI values are unavailable.",
                "leadership_focus": "Validate KPI calculation before using the executive summary.",
            },
            "metrics": {},
            "insights": {},
            "recommendations": {},
        }

    score_rows = []
    for key, item in kpis.items():
        value = safe_float(item.get("value"))
        score_rows.append({
            "key": key,
            "label": str(item.get("label", key)),
            "value": value,
            "display": str(item.get("display", value)),
            "status": str(item.get("status", "Monitoring")),
            "meaning": str(item.get("meaning", "")),
            "score": kpi_health_score(key, value),
        })

    scores_df = pd.DataFrame(score_rows)
    overall_score = safe_float(scores_df["score"].mean()) if not scores_df.empty else 0.0
    risk_count = int((scores_df["score"] < 70).sum()) if not scores_df.empty else 0
    high_risk_count = int((scores_df["score"] < 55).sum()) if not scores_df.empty else 0
    strongest = scores_df.sort_values("score", ascending=False).head(1).iloc[0].to_dict() if not scores_df.empty else {}
    weakest = scores_df.sort_values("score", ascending=True).head(1).iloc[0].to_dict() if not scores_df.empty else {}

    if high_risk_count >= 2 or overall_score < 55:
        status = {
            "status": "Critical KPI Concern",
            "severity": "critical",
            "summary": "Multiple KPI signals are below healthy thresholds and require executive attention.",
            "leadership_focus": "Prioritize the weakest KPIs and validate pressure relief actions before submission decisions.",
        }
    elif risk_count >= 2 or overall_score < 70:
        status = {
            "status": "Elevated KPI Risk",
            "severity": "elevated",
            "summary": "The KPI set contains multiple watch or risk signals for the selected period.",
            "leadership_focus": "Review KPI drivers and focus on the weakest operational signals first.",
        }
    elif risk_count == 1:
        status = {
            "status": "Watch Required",
            "severity": "watch",
            "summary": "Most KPIs are acceptable, but one signal requires monitoring.",
            "leadership_focus": "Track the weakest KPI and confirm it does not deteriorate in the next reporting window.",
        }
    elif overall_score >= 85:
        status = {
            "status": "Healthy Operating View",
            "severity": "relief",
            "summary": "The five guideline KPIs indicate a healthy operating view for the selected filter state.",
            "leadership_focus": "Maintain monitoring and document the stable operating conditions.",
        }
    else:
        status = {
            "status": "Stable Monitoring",
            "severity": "stable",
            "summary": "The five guideline KPIs support routine monitoring with no broad critical signal.",
            "leadership_focus": "Maintain standard KPI monitoring and watch for trend shifts.",
        }

    pressure_distribution = {}
    dominant_pressure_level = "Unclassified"
    dominant_driver = "Unclassified"

    if not df_filtered_daily.empty and "final_pressure_stress_level" in df_filtered_daily.columns:
        pressure_distribution = df_filtered_daily["final_pressure_stress_level"].fillna("Unclassified").astype(str).value_counts().to_dict()
        if pressure_distribution:
            dominant_pressure_level = max(pressure_distribution, key=pressure_distribution.get)

    if not df_filtered_daily.empty and "primary_pressure_driver" in df_filtered_daily.columns:
        drivers = df_filtered_daily["primary_pressure_driver"].fillna("Unclassified").astype(str).value_counts().to_dict()
        if drivers:
            dominant_driver = max(drivers, key=drivers.get)

    situation = (
        f"Overall KPI health score is {overall_score:.1f}/100. "
        f"Strongest signal is {strongest.get('label', 'Unavailable')} and weakest signal is {weakest.get('label', 'Unavailable')}."
    )
    risk = (
        f"{risk_count} KPI(s) are below the watch threshold and {high_risk_count} KPI(s) are in high-risk range. "
        f"Dominant pressure level is {dominant_pressure_level}."
    )
    diagnostic = (
        f"Metric toggles selected {len(selected_metrics)} diagnostic signal(s). KPI verdict remains anchored to the five guideline-required KPIs."
        if selected_metrics else
        "No metric toggles are selected for KPI diagnostics; Module 4 still evaluates the five guideline-required KPIs."
    )

    recommendations = {
        "leadership": status["leadership_focus"],
        "operations": f"Investigate the weakest KPI first: {weakest.get('label', 'Unavailable')}.",
        "planning": "Use KPI health, pressure distribution, and primary driver evidence to guide capacity, staffing, and discharge planning.",
        "risk_watch": f"Watch dominant pressure driver: {dominant_driver}. Confirm whether it aligns with backlog, volatility, or offset weakness.",
        "period_note": f"Module 4 evidence reflects the current {context.get('period_label', 'daily')} view and selected date range.",
    }

    return {
        "has_data": True,
        "granularity_context": context,
        "selected_metrics": selected_metrics,
        "status": status,
        "metrics": {
            "latest_date_display": safe_latest_date_display(df_filtered_daily),
            "period_count": len(df_filtered_daily),
            "overall_score": overall_score,
            "risk_count": risk_count,
            "high_risk_count": high_risk_count,
            "strongest_label": strongest.get("label", "Unavailable"),
            "strongest_score": safe_float(strongest.get("score", 0)),
            "weakest_label": weakest.get("label", "Unavailable"),
            "weakest_score": safe_float(weakest.get("score", 0)),
            "dominant_pressure_level": dominant_pressure_level,
            "dominant_driver": dominant_driver,
            "score_rows": score_rows,
        },
        "insights": {
            "situation": situation,
            "risk": risk,
            "diagnostic": diagnostic,
            "executive_summary": status["summary"],
        },
        "recommendations": recommendations,
    }


def render_module_command_header(
    prefix: str,
    module_number: str,
    title: str,
    subtitle: str,
    intelligence: dict[str, Any],
) -> None:
    """Render reusable command header for Module 2–4."""
    status = intelligence.get("status", {})
    metrics = intelligence.get("metrics", {})
    insights = intelligence.get("insights", {})
    context = intelligence.get("granularity_context", {})
    severity = hhs_safe_class_token(status.get("severity", "neutral"))
    status_label = hhs_escape_html(status.get("status", "Monitoring"))

    html = f"""
    <section class="hhs-{prefix}-command-shell hhs-{prefix}-status-{severity}">
        <div class="hhs-{prefix}-command-header">
            <div class="hhs-{prefix}-command-eyebrow">Module {hhs_escape_html(module_number)} · Production Healthcare Analytics Command Center</div>
            <div class="hhs-{prefix}-command-topline">
                <div class="hhs-{prefix}-command-title-block">
                    <h2 class="hhs-{prefix}-command-title">{hhs_escape_html(title)}</h2>
                    <p class="hhs-{prefix}-command-subtitle">{hhs_escape_html(subtitle)}</p>
                </div>
                <div class="hhs-{prefix}-status-badge">{status_label}</div>
            </div>
            <div class="hhs-{prefix}-command-summary">
                <div class="hhs-{prefix}-command-summary-main">{hhs_escape_html(status.get('summary', 'Status has been calculated from selected-period data.'))}</div>
                <div class="hhs-{prefix}-command-summary-support">{hhs_escape_html(insights.get('executive_summary', status.get('summary', 'Executive summary unavailable.')))}</div>
            </div>
            <div class="hhs-{prefix}-command-focus-row">
                <div class="hhs-{prefix}-focus-item"><span>Leadership Focus</span><strong>{hhs_escape_html(status.get('leadership_focus', 'Maintain monitoring.'))}</strong></div>
                <div class="hhs-{prefix}-focus-item"><span>Latest Period</span><strong>{hhs_escape_html(metrics.get('latest_date_display', 'Unavailable'))}</strong></div>
                <div class="hhs-{prefix}-focus-item"><span>Current View</span><strong>{hhs_escape_html(str(context.get('period_label', 'daily')).title())}</strong></div>
            </div>
        </div>
    </section>
    """
    hhs_render_html(html)


def render_module_evidence_grid(prefix: str, intelligence: dict[str, Any], cards: list[dict[str, str]]) -> None:
    """Render evidence cards for Module 2–4."""
    severity = hhs_safe_class_token(intelligence.get("status", {}).get("severity", "neutral"))
    items = "".join(
        f"""
        <article class="hhs-{prefix}-evidence-card hhs-{prefix}-status-{severity}">
            <span>{hhs_escape_html(card.get('label', 'Metric'))}</span>
            <strong>{hhs_escape_html(card.get('value', 'Unavailable'))}</strong>
            <em>{hhs_escape_html(card.get('meta', ''))}</em>
        </article>
        """
        for card in cards
    )
    hhs_render_html(f"<section class=\"hhs-{prefix}-evidence-grid\">{items}</section>")


def render_module_section_anchor(prefix: str, kicker: str, title: str) -> None:
    """Render section anchor for Module 2–4."""
    hhs_render_html(
        f"""
        <div class="hhs-{prefix}-section-anchor">
            <span>{hhs_escape_html(kicker)}</span>
            <strong>{hhs_escape_html(title)}</strong>
        </div>
        """
    )


def render_module_insight_block(
    prefix: str,
    intelligence: dict[str, Any],
    title: str,
    insight: str,
    recommendation: str,
    insight_label: str = "Dynamic Insight",
    recommendation_label: str = "Recommended Action",
) -> None:
    """Render insight and recommendation pair for Module 2–4."""
    severity = hhs_safe_class_token(intelligence.get("status", {}).get("severity", "neutral"))
    hhs_render_html(
        f"""
        <section class="hhs-{prefix}-intel-block hhs-{prefix}-status-{severity}">
            <div class="hhs-{prefix}-intel-title">{hhs_escape_html(title)}</div>
            <div class="hhs-{prefix}-intel-grid">
                <article class="hhs-{prefix}-insight-card"><span>{hhs_escape_html(insight_label)}</span><p>{hhs_escape_html(insight)}</p></article>
                <article class="hhs-{prefix}-reco-card"><span>{hhs_escape_html(recommendation_label)}</span><p>{hhs_escape_html(recommendation)}</p></article>
            </div>
        </section>
        """
    )


def render_module_explanation(prefix: str, title: str, what_it_shows: str, how_to_read: str, focus: str, intelligence: dict[str, Any]) -> None:
    """Render a concise how-to-read block."""
    severity = hhs_safe_class_token(intelligence.get("status", {}).get("severity", "neutral"))
    hhs_render_html(
        f"""
        <section class="hhs-{prefix}-explanation hhs-{prefix}-status-{severity}">
            <div class="hhs-{prefix}-explanation-title">{hhs_escape_html(title)}</div>
            <div class="hhs-{prefix}-explanation-grid">
                <article><span>What this shows</span><p>{hhs_escape_html(what_it_shows)}</p></article>
                <article><span>How to read it</span><p>{hhs_escape_html(how_to_read)}</p></article>
                <article><span>Decision focus</span><p>{hhs_escape_html(focus)}</p></article>
            </div>
        </section>
        """
    )


def render_module_decision_panel(prefix: str, intelligence: dict[str, Any], title_suffix: str) -> None:
    """Render executive decision-support closeout for Module 2–4."""
    severity = hhs_safe_class_token(intelligence.get("status", {}).get("severity", "neutral"))
    status = intelligence.get("status", {})
    insights = intelligence.get("insights", {})
    recommendations = intelligence.get("recommendations", {})
    status_label = hhs_escape_html(status.get("status", "Monitoring"))
    items = [
        ("Leadership Priority", recommendations.get("leadership", "Maintain leadership monitoring.")),
        ("Operations Monitoring", recommendations.get("operations", "Continue operational monitoring.")),
        ("Planning Action", recommendations.get("planning", "Maintain planning readiness.")),
        ("Risk Watch", recommendations.get("risk_watch", "Continue standard risk monitoring.")),
    ]
    item_html = "".join(
        f"<article><span>{hhs_escape_html(label)}</span><p>{hhs_escape_html(text)}</p></article>"
        for label, text in items
    )
    hhs_render_html(
        f"""
        <section class="hhs-{prefix}-decision-panel hhs-{prefix}-status-{severity}">
            <div class="hhs-{prefix}-decision-header"><span>Executive Decision Support</span><strong>{status_label} {hhs_escape_html(title_suffix)}</strong></div>
            <div class="hhs-{prefix}-decision-summary">{hhs_escape_html(insights.get('executive_summary', status.get('summary', 'Executive summary unavailable.')))}</div>
            <div class="hhs-{prefix}-decision-grid">{item_html}</div>
            <div class="hhs-{prefix}-period-note">{hhs_escape_html(recommendations.get('period_note', 'Recommendations reflect the selected filter state.'))}</div>
        </section>
        """
    )


def create_module2_load_comparison(df: pd.DataFrame, intelligence: dict[str, Any]) -> go.Figure:
    """Create Module 2 CBP vs HHS comparison chart."""
    if df.empty or not {"date", "cbp_custody", "hhs_care"}.issubset(df.columns):
        return empty_figure("CBP vs HHS comparison is unavailable because required columns are missing.")

    palette = get_status_palette(intelligence.get("status", {}).get("severity", "stable"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cbp_custody"], mode="lines", name="CBP Custody",
        line={"color": COLORS["slate"], "width": 3},
        hovertemplate="%{x|%b %d, %Y}<br>CBP Custody: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["hhs_care"], mode="lines", name="HHS Care",
        line={"color": palette["primary"], "width": 3},
        fill="tonexty", fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="%{x|%b %d, %Y}<br>HHS Care: %{y:,.0f}<extra></extra>",
    ))
    if "total_system_load" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["total_system_load"], mode="lines", name="Total System Load",
            line={"color": COLORS["navy"], "width": 2, "dash": "dot"},
            hovertemplate="%{x|%b %d, %Y}<br>Total Load: %{y:,.0f}<extra></extra>",
        ))
    return apply_command_center_chart_layout(
        fig,
        title="CBP vs HHS Responsibility Trend",
        yaxis_title="Children Under Care",
        height=430,
    )


def create_module2_share_donut(intelligence: dict[str, Any]) -> go.Figure:
    """Create latest CBP/HHS share donut using calculated intelligence."""
    metrics = intelligence.get("metrics", {})
    cbp = safe_float(metrics.get("latest_cbp"))
    hhs = safe_float(metrics.get("latest_hhs"))
    if cbp + hhs <= 0:
        return empty_figure("Latest care-load share is unavailable.")
    palette = get_status_palette(intelligence.get("status", {}).get("severity", "stable"))
    fig = go.Figure(data=[go.Pie(
        labels=["CBP Custody", "HHS Care"], values=[cbp, hhs], hole=0.64,
        marker={"colors": [COLORS["slate"], palette["primary"]]},
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    )])
    fig.update_layout(
        annotations=[{"text": "Latest<br>Responsibility", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 13, "color": COLORS["navy"]}}],
        showlegend=False,
    )
    return apply_command_center_chart_layout(fig, title="Latest Responsibility Split", height=360, show_legend=False)


def create_module2_share_shift(df: pd.DataFrame, intelligence: dict[str, Any]) -> go.Figure:
    """Create HHS and CBP share shift chart."""
    if df.empty or not {"date", "cbp_custody", "hhs_care"}.issubset(df.columns):
        return empty_figure("Responsibility shift view is unavailable.")
    chart_df = df.sort_values("date").copy()
    total = chart_df["cbp_custody"].fillna(0) + chart_df["hhs_care"].fillna(0)
    chart_df["HHS Share"] = np.where(total > 0, chart_df["hhs_care"] / total * 100, 0)
    chart_df["CBP Share"] = np.where(total > 0, chart_df["cbp_custody"] / total * 100, 0)
    palette = get_status_palette(intelligence.get("status", {}).get("severity", "stable"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["HHS Share"], mode="lines", name="HHS Share", line={"color": palette["primary"], "width": 3}, hovertemplate="%{x|%b %d, %Y}<br>HHS Share: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["CBP Share"], mode="lines", name="CBP Share", line={"color": COLORS["slate"], "width": 2}, hovertemplate="%{x|%b %d, %Y}<br>CBP Share: %{y:.1f}%<extra></extra>"))
    return apply_command_center_chart_layout(fig, title="Responsibility Share Shift", height=360, yaxis_title="Share of Total Load (%)")


def create_module3_flow_balance(df: pd.DataFrame, intelligence: dict[str, Any]) -> go.Figure:
    """Create Module 3 transfers vs discharges chart."""
    if df.empty or not {"date", "transfers_to_hhs", "hhs_discharged"}.issubset(df.columns):
        return empty_figure("Transfers vs discharges chart is unavailable because required columns are missing.")
    palette = get_status_palette(intelligence.get("status", {}).get("severity", "stable"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["transfers_to_hhs"], name="Transfers into HHS", marker={"color": palette["secondary"]}, hovertemplate="%{x|%b %d, %Y}<br>Transfers: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=df["date"], y=df["hhs_discharged"], name="Discharges from HHS", marker={"color": COLORS["teal"]}, hovertemplate="%{x|%b %d, %Y}<br>Discharges: %{y:,.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    return apply_command_center_chart_layout(fig, title="Transfers vs Discharges Flow Balance", yaxis_title="Children", height=420)


def create_module3_net_pressure(df: pd.DataFrame, intelligence: dict[str, Any]) -> go.Figure:
    """Create Module 3 net intake pressure chart."""
    if df.empty or "net_daily_intake" not in df.columns:
        return empty_figure("Net intake pressure chart is unavailable because required columns are missing.")
    net_intake = pd.to_numeric(df["net_daily_intake"], errors="coerce").fillna(0)
    colors = np.where(net_intake > 0, COLORS["orange"], np.where(net_intake < 0, COLORS["teal"], COLORS["slate_light"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=net_intake, name="Net Intake", marker={"color": colors}, hovertemplate="%{x|%b %d, %Y}<br>Net Intake: %{y:,.0f}<extra></extra>"))
    if "rolling_7d_avg_net_intake" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_7d_avg_net_intake"], mode="lines", name="Rolling Net Intake Avg", line={"color": COLORS["navy"], "width": 2}, hovertemplate="%{x|%b %d, %Y}<br>Rolling Avg: %{y:,.1f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["slate"], annotation_text="Balanced flow", annotation_position="top left")
    return apply_command_center_chart_layout(fig, title="Net Intake Pressure and Relief Signal", yaxis_title="Transfers − Discharges", height=420)


def create_module3_backlog_trend(df: pd.DataFrame, intelligence: dict[str, Any]) -> go.Figure:
    """Create Module 3 backlog severity chart."""
    if df.empty or "backlog_severity_score" not in df.columns:
        return empty_figure("Backlog severity trend is unavailable because required columns are missing.")
    severity = pd.to_numeric(df["backlog_severity_score"], errors="coerce").fillna(0)
    palette = get_status_palette(intelligence.get("status", {}).get("severity", "stable"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=severity, mode="lines+markers", name="Backlog Severity", line={"color": palette["primary"], "width": 3}, marker={"size": 5, "color": palette["secondary"]}, fill="tozeroy", fillcolor="rgba(249, 115, 22, 0.14)", hovertemplate="%{x|%b %d, %Y}<br>Backlog Severity: %{y:.2f}<extra></extra>"))
    return apply_command_center_chart_layout(fig, title="Backlog Severity Accumulation", height=360, yaxis_title="Severity Score")


def create_module3_pressure_relief_strip(df: pd.DataFrame, intelligence: dict[str, Any]) -> go.Figure:
    """Create pressure/relief strip for Module 3."""
    if df.empty or "net_daily_intake" not in df.columns:
        return empty_figure("Pressure/relief strip is unavailable because required columns are missing.")
    heat_df = df[["date", "net_daily_intake"]].copy()
    heat_df["signal"] = np.select([heat_df["net_daily_intake"] > 0, heat_df["net_daily_intake"] < 0], [1, -1], default=0)
    fig = go.Figure(data=go.Heatmap(
        x=heat_df["date"], y=["Flow Signal"] * len(heat_df), z=[heat_df["signal"]],
        colorscale=[[0.0, COLORS["teal"]], [0.49, COLORS["teal"]], [0.50, COLORS["slate_light"]], [0.51, COLORS["orange"]], [1.0, COLORS["red"]]],
        showscale=True,
        colorbar={"title": "Signal", "tickvals": [-1, 0, 1], "ticktext": ["Relief", "Balanced", "Pressure"]},
        hovertemplate="%{x|%b %d, %Y}<br>Signal: %{z}<extra></extra>",
    ))
    return apply_command_center_chart_layout(fig, title="Pressure and Relief Window Strip", height=280, show_legend=False)


def create_module4_kpi_health_ranking(intelligence: dict[str, Any]) -> go.Figure:
    """Create KPI health ranking chart where higher score is better."""
    rows = intelligence.get("metrics", {}).get("score_rows", [])
    if not rows:
        return empty_figure("KPI health ranking is unavailable.")
    df = pd.DataFrame(rows).sort_values("score", ascending=True)
    colors = [COLORS["red"] if score < 55 else COLORS["orange"] if score < 70 else COLORS["teal"] if score >= 85 else COLORS["blue"] for score in df["score"]]
    fig = go.Figure(data=[go.Bar(x=df["score"], y=df["label"], orientation="h", marker={"color": colors}, text=[f"{score:.0f}" for score in df["score"]], textposition="outside", hovertemplate="%{y}<br>Health Score: %{x:.1f}/100<extra></extra>")])
    fig.update_xaxes(range=[0, 105])
    return apply_command_center_chart_layout(fig, title="Guideline KPI Health Ranking", height=390, xaxis_title="Health Score / 100", show_legend=False)


def create_module4_pressure_distribution(df: pd.DataFrame) -> go.Figure:
    """Create Module 4 pressure distribution donut."""
    if df.empty or "final_pressure_stress_level" not in df.columns:
        return empty_figure("Pressure distribution is unavailable because required columns are missing.")
    distribution = df["final_pressure_stress_level"].fillna("Unclassified").astype(str).value_counts().reset_index()
    distribution.columns = ["Pressure Level", "Periods"]
    colors = [pressure_level_color(level) for level in distribution["Pressure Level"]]
    fig = go.Figure(data=[go.Pie(labels=distribution["Pressure Level"], values=distribution["Periods"], hole=0.58, marker={"colors": colors}, textinfo="label+percent", hovertemplate="%{label}<br>Periods: %{value:,.0f}<br>Share: %{percent}<extra></extra>")])
    return apply_command_center_chart_layout(fig, title="Pressure Stress Distribution", height=390, show_legend=False)


def create_module4_driver_distribution(df: pd.DataFrame) -> go.Figure:
    """Create Module 4 primary pressure driver chart."""
    if df.empty or "primary_pressure_driver" not in df.columns:
        return empty_figure("Primary pressure driver distribution is unavailable because required columns are missing.")
    driver_df = df["primary_pressure_driver"].fillna("Unclassified").astype(str).value_counts().head(8).sort_values().reset_index()
    driver_df.columns = ["Primary Pressure Driver", "Periods"]
    fig = go.Figure(data=[go.Bar(x=driver_df["Periods"], y=driver_df["Primary Pressure Driver"], orientation="h", marker={"color": COLORS["blue"]}, hovertemplate="%{y}<br>Periods: %{x:,.0f}<extra></extra>")])
    return apply_command_center_chart_layout(fig, title="Primary Pressure Driver Evidence", height=380, xaxis_title="Periods", show_legend=False)


def render_module_2_command_center(
    df_visual: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> None:
    """Render Module 2 production CBP vs HHS command center."""
    intelligence = calculate_module2_intelligence(df_visual, selected_granularity, selected_metrics)
    render_module_command_header(
        "m2",
        "2",
        "CBP vs HHS Load Comparison Analysis",
        "Filter-aware responsibility-split intelligence for CBP custody, HHS care concentration, handoff balance, and downstream care-load planning.",
        intelligence,
    )
    if not intelligence.get("has_data"):
        return
    m = intelligence.get("metrics", {})
    cards = [
        {"label": "Latest CBP Custody", "value": format_compact_number(m.get("latest_cbp")), "meta": f"{format_pct(m.get('current_cbp_share'), 1)} of latest load"},
        {"label": "Latest HHS Care", "value": format_compact_number(m.get("latest_hhs")), "meta": f"{format_pct(m.get('current_hhs_share'), 1)} of latest load"},
        {"label": "HHS / CBP Ratio", "value": "∞" if m.get("hhs_to_cbp_ratio") == float("inf") else format_ratio(m.get("hhs_to_cbp_ratio")), "meta": "Downstream care concentration"},
        {"label": "HHS Share Shift", "value": format_pp(m.get("hhs_share_delta_pp")), "meta": "Latest vs selected-period average"},
        {"label": "Load Gap", "value": format_signed_number(m.get("load_gap")), "meta": "HHS care minus CBP custody"},
        {"label": "Filtered Periods", "value": format_compact_number(m.get("period_count")), "meta": "Data points in current view"},
    ]
    render_module_evidence_grid("m2", intelligence, cards)
    render_module_section_anchor("m2", "Responsibility Intelligence", "CBP vs HHS Care Load Movement")
    st.plotly_chart(create_module2_load_comparison(df_visual, intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    render_module_insight_block("m2", intelligence, "Responsibility Split Intelligence", intelligence.get("insights", {}).get("situation", "Insight unavailable."), intelligence.get("recommendations", {}).get("operations", "Continue monitoring."), "Load Concentration Insight", "Operations Recommendation")
    col_1, col_2 = st.columns([1.0, 1.35], gap="large")
    with col_1:
        st.plotly_chart(create_module2_share_donut(intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    with col_2:
        st.plotly_chart(create_module2_share_shift(df_visual, intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    render_module_insight_block("m2", intelligence, "Handoff Shift and Diagnostic Context", intelligence.get("insights", {}).get("shift", "Shift insight unavailable."), intelligence.get("insights", {}).get("diagnostic", "Diagnostic note unavailable."), "Share Shift Insight", "Metric Toggle Context")
    render_module_explanation("m2", "How to Read CBP vs HHS Load Comparison", "This module compares upstream CBP custody load with downstream HHS care load for the selected filter state.", "A higher HHS share means most care responsibility is downstream; a rising CBP share can indicate intake-side or transfer-handoff pressure.", "Focus on responsibility shifts, HHS/CBP ratio, and widening load gaps because these affect care-capacity planning.", intelligence)
    render_module_decision_panel("m2", intelligence, "Responsibility View")


def render_module_3_command_center(
    df_visual: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> None:
    """Render Module 3 production net intake and backlog command center."""
    intelligence = calculate_module3_intelligence(df_visual, selected_granularity, selected_metrics)
    render_module_command_header(
        "m3",
        "3",
        "Net Intake & Backlog Pressure Analysis",
        "Flow-balance intelligence for transfers into HHS, discharges from HHS, net intake pressure, relief periods, and backlog accumulation.",
        intelligence,
    )
    if not intelligence.get("has_data"):
        return
    m = intelligence.get("metrics", {})
    context = intelligence.get("granularity_context", {})
    cards = [
        {"label": "Transfers into HHS", "value": format_compact_number(m.get("total_transfers")), "meta": "Total selected-period inflow"},
        {"label": "Discharges from HHS", "value": format_compact_number(m.get("total_discharges")), "meta": "Total selected-period relief"},
        {"label": "Net Intake Total", "value": format_signed_number(m.get("net_total")), "meta": "Transfers minus discharges"},
        {"label": "Avg Net Intake", "value": format_compact_number(m.get("avg_net"), 1), "meta": context.get("average_label", "average per period")},
        {"label": "Discharge Offset", "value": format_ratio(m.get("offset_ratio")), "meta": "Discharges divided by transfers"},
        {"label": "Pressure Period Share", "value": format_pct(m.get("pressure_share"), 1), "meta": f"Longest pressure streak: {format_compact_number(m.get('longest_pressure'))}"},
    ]
    render_module_evidence_grid("m3", intelligence, cards)
    render_module_section_anchor("m3", "Flow Balance Intelligence", "Transfers, Discharges, and Net Pressure")
    st.plotly_chart(create_module3_flow_balance(df_visual, intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    render_module_insight_block("m3", intelligence, "Flow Balance Intelligence", intelligence.get("insights", {}).get("situation", "Insight unavailable."), intelligence.get("recommendations", {}).get("operations", "Continue monitoring."), "Pressure / Relief Insight", "Operations Recommendation")
    st.plotly_chart(create_module3_net_pressure(df_visual, intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    col_1, col_2 = st.columns([1.15, 1.0], gap="large")
    with col_1:
        st.plotly_chart(create_module3_backlog_trend(df_visual, intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    with col_2:
        st.plotly_chart(create_module3_pressure_relief_strip(df_visual, intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    render_module_insight_block("m3", intelligence, "Backlog and Offset Intelligence", intelligence.get("insights", {}).get("offset", "Offset insight unavailable."), intelligence.get("recommendations", {}).get("planning", "Maintain planning readiness."), "Discharge Offset Insight", "Planning Recommendation")
    render_module_explanation("m3", "How to Read Net Intake & Backlog Trends", "This module tracks whether transfers into HHS exceed discharges from HHS and whether pressure is becoming sustained backlog.", "Positive net intake indicates pressure buildup; negative net intake indicates relief; repeated positive periods indicate backlog accumulation risk.", "Focus on discharge offset ratio, pressure-period share, longest pressure streak, and backlog severity because these signals drive capacity stress.", intelligence)
    render_module_decision_panel("m3", intelligence, "Flow-Balance View")


def render_module_4_command_center(
    kpis: dict[str, dict[str, Any]],
    df_filtered_daily: pd.DataFrame,
    selected_granularity: str,
    selected_metrics: list[str] | None = None,
) -> None:
    """Render Module 4 production KPI executive command center."""
    intelligence = calculate_module4_intelligence(kpis, df_filtered_daily, selected_granularity, selected_metrics)
    render_module_command_header(
        "m4",
        "4",
        "KPI Executive Summary and Health Ranking",
        "Executive interpretation of the five guideline-required KPIs, pressure distribution, primary drivers, and operational decision priorities.",
        intelligence,
    )
    if not intelligence.get("has_data"):
        return
    m = intelligence.get("metrics", {})
    cards = [
        {"label": "Overall KPI Health", "value": f"{safe_float(m.get('overall_score')):.1f}/100", "meta": "Composite executive score"},
        {"label": "Strongest KPI", "value": str(m.get("strongest_label", "Unavailable")), "meta": f"Score: {safe_float(m.get('strongest_score')):.0f}/100"},
        {"label": "Weakest KPI", "value": str(m.get("weakest_label", "Unavailable")), "meta": f"Score: {safe_float(m.get('weakest_score')):.0f}/100"},
        {"label": "Watch KPIs", "value": format_compact_number(m.get("risk_count")), "meta": "Below 70 health score"},
        {"label": "Dominant Pressure Level", "value": str(m.get("dominant_pressure_level", "Unclassified")), "meta": "Most frequent daily pressure state"},
        {"label": "Primary Driver", "value": str(m.get("dominant_driver", "Unclassified")), "meta": "Most frequent pressure driver"},
    ]
    render_module_evidence_grid("m4", intelligence, cards)
    render_module_section_anchor("m4", "Executive KPI Intelligence", "Guideline KPI Health Ranking")
    st.plotly_chart(create_module4_kpi_health_ranking(intelligence), use_container_width=True, config=PLOTLY_CONFIG)
    render_module_insight_block("m4", intelligence, "KPI Health Intelligence", intelligence.get("insights", {}).get("situation", "Insight unavailable."), intelligence.get("recommendations", {}).get("operations", "Continue monitoring."), "Executive KPI Insight", "Operations Recommendation")
    col_1, col_2 = st.columns([1.0, 1.15], gap="large")
    with col_1:
        st.plotly_chart(create_module4_pressure_distribution(df_filtered_daily), use_container_width=True, config=PLOTLY_CONFIG)
    with col_2:
        st.plotly_chart(create_module4_driver_distribution(df_filtered_daily), use_container_width=True, config=PLOTLY_CONFIG)
    render_module_insight_block("m4", intelligence, "Pressure Driver and Risk Intelligence", intelligence.get("insights", {}).get("risk", "Risk insight unavailable."), intelligence.get("recommendations", {}).get("risk_watch", "Continue risk monitoring."), "Risk Summary", "Risk Watch")
    render_module_explanation("m4", "How to Read KPI Executive Summary", "This module converts the five required KPIs into executive health signals and connects them with pressure distribution and driver evidence.", "Higher KPI health scores indicate stronger operating conditions; watch low-scoring KPIs, dominant pressure levels, and frequent pressure drivers.", "Focus on the weakest KPI first, then confirm whether pressure distribution and primary drivers explain the issue.", intelligence)
    render_module_decision_panel("m4", intelligence, "KPI Operating View")
    st.markdown("### KPI Interpretation Reference")
    st.dataframe(build_kpi_interpretation_table(kpis), use_container_width=True, hide_index=True)

# ============================================================
# 16. Main Application
# ============================================================

def main() -> None:
    """Run the Streamlit healthcare analytics dashboard."""
    load_css(CSS_PATH)

    main_path = first_existing_path(DATA_FILE_CANDIDATES["main"])

    if main_path is None:
        st.error(
            "The final dashboard dataset was not found. Expected file: "
            "`final_pressure_stress_classification_dataset.csv` inside `notebooks/data/processed/`."
        )
        st.stop()

    raw_df = load_csv_safely(str(main_path))
    df = prepare_dashboard_dataset(raw_df)

    if df.empty:
        st.error("The final dashboard dataset is empty or could not be loaded.")
        st.stop()

    if not required_columns_available(df):
        st.error(
            "The dataset is missing one or more core guideline columns required for the dashboard: "
            "Date, CBP custody, HHS care, transfers into HHS, and HHS discharges."
        )
        st.stop()

    df_kpi_cards = load_optional_dataset("kpi_cards")
    df_final_kpi_summary = load_optional_dataset("final_kpi_summary")
    _ = df_kpi_cards, df_final_kpi_summary

    selected_date_range, selected_granularity, selected_metrics = render_sidebar(df)

    df_filtered_daily = apply_date_filter(df, selected_date_range)

    if df_filtered_daily.empty:
        st.warning("No records are available for the selected date range.")
        st.stop()

    df_visual = aggregate_by_granularity(df_filtered_daily, selected_granularity)

    if df_visual.empty:
        st.warning("No records are available after applying the selected time granularity.")
        st.stop()

    # KPIs must be calculated from the same dataset used by the dashboard visuals.
    # This makes KPI cards fully connected to Date Range + Time Granularity filters.
    kpis = calculate_kpis(df_visual)
    selected_metric_kpis = calculate_selected_metric_kpis(df_visual, selected_metrics) 

    render_header()

    st.markdown("## Key Performance Indicators (KPIs)")
    render_guideline_kpi_cards(kpis, selected_granularity)
    
    st.markdown(
    """
    <div class="hhs-module-nav-header">
        <div class="hhs-module-nav-title">Dashboard Modules</div>
        <div class="hhs-module-nav-description">
            Navigate through the guideline-required analytical modules for system load,
            CBP/HHS comparison, intake pressure, backlog trends, and KPI interpretation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

    tab_1, tab_2, tab_3, tab_4 = st.tabs(
        [
            "System Load Overview",
            "CBP vs HHS Load Comparison",
            "Net Intake & Backlog Trends",
            "KPI Summary Cards",
        ]
    )

    with tab_1:
        render_module_1_command_center(
            df_visual=df_visual,
            df_filtered_daily=df_filtered_daily,
            selected_granularity=selected_granularity,
            selected_metrics=selected_metrics,
        )

    with tab_2:
        render_module_2_command_center(
            df_visual=df_visual,
            selected_granularity=selected_granularity,
            selected_metrics=selected_metrics,
        )

    with tab_3:
        render_module_3_command_center(
            df_visual=df_visual,
            selected_granularity=selected_granularity,
            selected_metrics=selected_metrics,
        )

    with tab_4:
        render_module_4_command_center(
            kpis=kpis,
            df_filtered_daily=df_filtered_daily,
            selected_granularity=selected_granularity,
            selected_metrics=selected_metrics,
        )

    st.markdown("## Filtered Capacity & Pressure Records")
    st.caption(
        "Curated analytical records for the selected dashboard scope. "
        "This table is filtered by date range and reflects the project guideline metrics only."
    )

    analytical_table = build_filtered_records_table(df_filtered_daily)

    st.dataframe(
        analytical_table,
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.caption(
        "Dashboard scope: system capacity, care load monitoring, inflow/outflow balance, "
        "backlog pressure, and discharge relief analytics for the UAC care pipeline."
    )


if __name__ == "__main__":
    main()
    
    
# -------------------------
# FOOTER SECTION
# -------------------------
st.markdown("---")

st.markdown("### 📌 Project Information & Credits")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
**👨‍💻 Developed by:** Mohit Gupta
  
**🎯 Role:** Data Analyst Intern
        """
    )

with c2:
    st.markdown(
        """
**📊 Project:** System Capacity & Care Load Analytics for Unaccompanied Children

**🏢 Organization:** Unified Mentor Pvt. Ltd.
        """
    )

with c3:
    st.markdown(
        """
**👨‍🏫 Mentor:** Saiprasad Kagne  

**📅 Year:** 2026
        """
    )

st.markdown(
    """
<div style="
    text-align: center;
    margin-top: 10px;
    color: #6b563d;
    font-size: 14px;
    font-weight: 600;
">
    💡 Built using Python, Pandas, Plotly & Streamlit
</div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)