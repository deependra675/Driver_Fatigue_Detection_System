""" automated unit tests for offline resilience and spooling recovery"""
import json
import os
import time
from edge_telemetry_dispatcher import ResilientEdgeDispatcher

TEST_SPOOL = "test_spool.jsonl"

def test_offline_network_drop_spools_to_disk():
    # clean up any leftover test files
    if os.path.exists(TEST_SPOOL):
        os.remove(TEST_SPOOL)

    # point to an unreachable port to simulate an edge network failure
    dead_endpoint = "http://127.0.0.1:9999/api/v1/telemetry/alert"
    dispatcher = ResilientEdgeDispatcher(
        api_endpoint=dead_endpoint, spool_path=TEST_SPOOL, timeout_sec=0.2
    )
    payload = {
        "driver_id": 901,
        "vehicle_category": "Spool Test Van",
        "ear_score": 0.12,
        "severity_score": 0.85,
        "consecutive_closed_frames": 8,
        "alert_type": "MICROSLEEP_CRITICAL",
    }
    # main cv loop enqueues without blocking
    queued = dispatcher.enqueue_alert(payload)
    assert queued is True

    # give background thread time to fail http and write to disk
    time.sleep(0.6)
    dispatcher.shutdown()

    # verify the disk buffer caught and preserved the data
    assert os.path.exists(TEST_SPOOL)
    with open(TEST_SPOOL, "r", encoding="utf-8") as f:
        saved_lines = [line.strip() for line in f if line.strip()]

    assert len(saved_lines) == 1
    saved_data = json.loads(saved_lines[0])
    assert saved_data["driver_id"] == 901
    assert saved_data["alert_type"] == "MICROSLEEP_CRITICAL"

    # clean up artifact
    if os.path.exists(TEST_SPOOL):
        os.remove(TEST_SPOOL)