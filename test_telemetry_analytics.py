"""
unit and boundary tests for the vectorized time series analytics engine."""
from datetime import datetime, timedelta
import pandas as pd
import pytest

from telemetry_analytics import(
    compute_temporal_risk_profile,
    generate_driver_risk_summary
)

def test_rolling_window_burst_and_decay():
    base_time = datetime(2026, 9, 22, 10, 0, 0)

    sample_data = {
        "incident_id": [1, 2, 3, 4],
        "driver_id": [101, 101, 101, 101],
        "ear_score": [0.20, 0.18, 0.14, 0.28],
        "severity_score": [0.70, 0.75, 0.80, 0.30],
        "alert_type": [
            "DROWSY_WARNING",
            "DROWSY_WARNING",
            "MICROSLEEP_CRITICAL",
            "DROWSY_WARNING"
        ],
        "recorded_at": [
            base_time,
            base_time + timedelta(minutes=2),
            base_time + timedelta(minutes=4),
            base_time + timedelta(minutes=34)
        ]
    }
    df = pd.DataFrame(sample_data)
    result = compute_temporal_risk_profile(df, rolling_window="15min")

    # at event index 0: 1 item, severity 0.70
    assert result.loc[0, "rolling_incident_count"] == 1
    assert result.loc[0, "rolling_mean_severity"] == 0.70
    assert bool(result.loc[0, "acute_risk_flag"]) is False

    # at event index 2: 3 items inside 4 mins (window is 15 min)
    assert result.loc[2, "rolling_incident_count"] == 3
    assert result.loc[2, "rolling_mean_severity"] == 0.75
    assert result.loc[2, "rolling_min_ear"] == 0.14
    assert bool(result.loc[2, "acute_risk_flag"]) is True

    # at event index 3; earlier events aged out (>15 min gap)
    assert result.loc[3, "rolling_incident_count"] == 1
    assert result.loc[3, "rolling_mean_severity"] == 0.30
    assert bool(result.loc[3, "acute_risk_flag"]) is False

def test_driver_summary_serialilzation():
    base_time = datetime(2026, 9, 22, 8, 0, 0)
    sample_data = {
        "incident_id": [10, 11, 12],
        "driver_id": [202, 202, 202],
        "ear_score": [0.15, 0.12, 0.10],
        "severity_score": [0.80, 0.85, 0.90],
        "alert_type": ["MICROSLEEP_CRITICAL"] * 3,
        "recorded_at": [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2)
        ]
    }
    df = pd.DataFrame(sample_data)
    processed_df = compute_temporal_risk_profile(df, rolling_window="15min")
    summary = generate_driver_risk_summary(processed_df)

    assert summary["driver_id"] == 202
    assert summary["total_incidents"] == 3
    assert summary["has_acute_risk_history"] is True
    assert summary["peak_burst_density"] == 3
    assert summary["latest_mean_severity"] == 0.85
    assert len(summary["timeline"]) == 3
