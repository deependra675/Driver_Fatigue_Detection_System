""" End to End system integration tests for edge CV telemetry gatway."""
import sqlite3
import pytest
from fastapi.testclient import TestClient
from friday_telemetry_api import app, DB_FILE

client = TestClient(app)

# test check if the server is awake and responding
def test_health_check_endpoint():
    # send a get request to /health
    response = client.get("/health")

    # did the server say 200 OK?
    assert response.status_code == 200
    # convert JSON text answer into a python dict
    data = response.json()
    # did the answer contain {"status": "healthy"}?
    assert data["status"] == "healthy"
    # did it tell us how many records are in the database?
    assert "persisted_records" in data

# test check the happy path (send clean data -> verify it saved)
def test_valid_telemetry_ingestion_and_persistence():
    payload = {
        "driver_id": 404,
        "vehicle_category": "Logistics Van",
        "ear_score": 0.145,
        "severity_score": 0.88,
        "consecutive_closed_frames": 7,
        "alert_type": "MICROSLEEP_CRITICAL",
    }

    # send the alert to fastapi
    response = client.post("/api/v1/telemetry/alert", json=payload)
    # check if the api created the record (201 mean created)
    assert response.status_code == 201

    resp_data = response.json()
    assert resp_data["status"] == "persisted_to_db"

    # grab the newly generated database ID returned by the api
    record_id = resp_data["record_id"]

    # open the actual sqlite db file directly on the disk to verify
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # look up the row that the api claims it just saved
        cur.execute(
            "SELECT * FROM fatigue_incident_logs WHERE incident_id = ?;",
            (record_id,),
        )
        row = cur.fetchone()

    # verify the database rerally wrote the exact values we sent
    assert row is not None
    assert row["driver_id"] == 404
    assert row["vehicle_category"] == "Logistics Van"
    assert row["consecutive_closed_frames"] == 7
    assert row["alert_type"] == "MICROSLEEP_CRITICAL"

# check first defense line 
def test_contract_validation_rejects_invalid_severity():
    # rule in our code: microsleep critical must have serverity >= 0.70
    # here we send 0.50 (nonsense data)
    invalid_payload = {
        "driver_id": 404,
        "vehicle_category": "Logistics Van",
        "ear_score": 0.12,
        "severity_score": 0.50,
        "consecutive_closed_frames": 6,
        "alert_type": "MICROSLEEP_CRITICAL"
    }
    # send it to the api
    response = client.post("/api/v1/telemetry/alert", json=invalid_payload)
    # 422 means: unprocessable entity, fastapi caught the bad logic and rejects
    assert response.status_code == 422

# test check second defense line
def test_database_check_constraint_direct():
   # what if a buggy script tries to insert corrupted data directly into sqllite
   # bypassing fastapi entirely?
   with sqlite3.connect(DB_FILE) as conn:
       cur = conn.cursor()
       with pytest.raises(sqlite3.IntegrityError):
            cur.execute("""
                INSERT INTO fatigue_incident_logs (
                    driver_id, vehicle_category, ear_score,
                    severity_score, consecutive_closed_frames,
                    alert_type, recorded_at
                ) VALUES (999, 'Test Truck', -0.5, 0.80, 5, 'DROWSY_WARNING', '2026-09-19T10:00:00');
            """)
            conn.commit()

def test_batch_telemetry_ingestion_and_atomicity():
    batch_payload = [
        {
            "driver_id": 801,
            "vehicle_category": "Freight Hauler",
            "ear_score": 0.12,
            "severity_score": 0.85,
            "consecutive_closed_frames": 9,
            "alert_type": "MICROSLEEP_CRITICAL",
        },
        {
            "driver_id": 802,
            "vehicle_category": "Delivery Van",
            "ear_score": 0.21,
            "severity_score": 0.40,
            "consecutive_closed_frames": 3,
            "alert_type": "DROWSY_WARNING",
        },        
        {
            "driver_id": 803,
            "vehicle_category": "Cab",
            "ear_score": 0.25,
            "severity_score": 0.30,
            "consecutive_closed_frames": 0,
            "alert_type": "YAWN_DETECTED",
        },
    ]

    # successful bulk ingestion
    response = client.post("/api/v1/telemetry/batch", json=batch_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "batch_persisted_to_db"
    assert data["records_inserted"] == 3

    # empty list rejection (400 bad request)
    empty_response = client.post("/api/v1/telemetry/batch", json=[])
    assert empty_response.status_code == 400

    # validation rejection: if any payload is corrupt, the entire batch aborts.
    corrupt_batch = [
        {
            "driver_id": 804,
            "vehicle_category": "Van",
            "ear_score": 0.10,
            "severity_score": 0.90,
            "consecutive_closed_frames": 6,
            "alert_type": "MICROSLEEP_CRITICAL",
        },
        {
            "driver_id": 805,
            "vehicle_category": "Van",
            "ear_score": 0.10,
            "severity_score": 0.20,
            "consecutive_closed_frames": 6,
            "alert_type": "MICROSLEEP_CRITICAL",
        },        
    ]
    corrupt_response = client.post("/api/v1/telemetry/batch", json=corrupt_batch)
    assert corrupt_response.status_code == 422