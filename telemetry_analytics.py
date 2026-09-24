"""
vectorized time series analytivs engine for driver fatigue telemetry.
extracts historical incident logs from SQLite and computes sliding temporal risk profiles using pandas rolling time based windows.
"""
import sqlite3
from typing import Optional, Dict, Any, List
import pandas as pd

def load_driver_telemetry_df(db_path: str, driver_id: int) -> Optional[pd.DataFrame]:
    query = """
        SELECT incident_id, driver_id, vehicle_category, ear_score,
               severity_score, consecutive_closed_frames, alert_type, recorded_at
        FROM fatigue_incident_logs
        WHERE driver_id = ?
        ORDER BY recorded_at ASC;
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(driver_id,),
            parse_dates=["recorded_at"]
        )
    if df.empty:
        return None

    # enforce UTC/clean datetime conversion and ascending sort
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").reset_index(drop=True)
    return df

def compute_temporal_risk_profile(
    df: pd.DataFrame,
    rolling_window: str = "15min"
) -> pd.DataFrame:
    if df.empty:
        return df
    df_proc = df.copy()

    # Time-based rolling windows require a DatetimeIndex
    df_indexed = df_proc.set_index("recorded_at")

    # Construct the continuous time rolling object
    rolling_obj = df_indexed.rolling(rolling_window)

    df_proc["rolling_mean_severity"] = (
        rolling_obj["severity_score"].mean().values.round(3)
    )
    df_proc["rolling_incident_count"] = (
        rolling_obj["incident_id"].count().values.astype(int)
    )
    # Fix: compute actual rolling minimum EAR score
    df_proc["rolling_min_ear"] = (
        rolling_obj["ear_score"].min().values.round(3)
    )
    # Acute risk condition: 3+ incidents inside the window with severity >= 0.65
    df_proc["acute_risk_flag"] = (
        (df_proc["rolling_incident_count"] >= 3) &
        (df_proc["rolling_mean_severity"] >= 0.65)
    )
    return df_proc

def generate_driver_risk_summary(df_proc: pd.DataFrame) -> Dict[str, Any]:
    if df_proc.empty:
        return {
            "driver_id": None,
            "total_incidents": 0,
            "has_acute_risk_history": False,
            "peak_burst_density": 0,
            "latest_mean_severity": 0.0,
            "timeline": []
        }
    latest_row = df_proc.iloc[-1]
    return {
        "driver_id": int(latest_row["driver_id"]),
        "total_incidents": int(len(df_proc)),
        "has_acute_risk_history": bool(df_proc["acute_risk_flag"].any()),
        "peak_burst_density": int(df_proc["rolling_incident_count"].max()),
        "latest_mean_severity": float(latest_row["rolling_mean_severity"]),
        "timeline": df_proc.to_dict(orient="records")
    }