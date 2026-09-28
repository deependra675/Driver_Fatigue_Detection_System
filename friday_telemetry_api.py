""" high-throughput telemetry ingestion microservice.
provides validated REST endpoints for edge fatigue events using FASTAPI & Pydantic.
"""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from datetime import datetime, timezone
from typing import List, Optional
import uvicorn
import sqlite3
# pydantic is our security guard, it checks incoming data to make sure it matches the rules we want.
# field lets us set extra rules like min lenth and field_validator.
# uvicorn is the acutual server engine that runs our fastapi app so it can listen for web traffic.

DB_FILE = "fleet_telemetry.db"
DB_PATH = DB_FILE
# we create a variable holding the name of our database file. this file will be created automatically in our folder to save the logs.
def init_db():       # defines a func that sets up our database tables when app starts.
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        # high concurrency pragmas
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA busy_timeout = 5000;")
        # DDL table with CHECK constraints
        cur.execute("""
                CREATE TABLE IF NOT EXISTS fatigue_incident_logs ( 
                    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    driver_id INTEGER NOT NULL,
                    vehicle_category TEXT NOT NULL,
                    ear_score REAL NOT NULL CHECK (ear_score >= 0.0 AND ear_score <= 1.0),
                    severity_score REAL NOT NULL CHECK (severity_score >= 0.0 AND severity_score <= 1.0),
                    consecutive_closed_frames INTEGER NOT NULL CHECK (consecutive_closed_frames >= 0),
                    alert_type TEXT NOT NULL CHECK (alert_type IN ('DROWSY_WARNING', 'MICROSLEEP_CRITICAL', 'YAWN_DETECTED')),
                    recorded_at TEXT NOT NULL
                );
            """)
        # Compund index for driver query speed
        cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_driver_timestamp 
                ON fatigue_incident_logs (driver_id, recorded_at DESC);
            """)
        conn.commit() # saves our database changes permanently
init_db()

app = FastAPI(  # this creates our web application instance
    title="Fleet Driver Telemetry Ingestion Service",
    version="1.0.0",
    description="Validates and captures real-time driver fatigue telemetry."
)

# --- Pydantic data contract ---
class FatigueAlertPayload(BaseModel): # defines exact structure of the JSON data we expect cars to send us.
    driver_id: int = Field(..., gt=0, description="Unique driver identifier") # ... means the field is required, gt=0 means the driverID must be greater that zero.
    vehicle_category: str = Field(..., min_length=2, max_length=50) # min_length/max_length/ge/le are built in pydantic rules that automatically reject invalid numbers or text before they ever reach our database.
    ear_score: float = Field(..., ge=0.0, le=1.0, description="Eye Aspect Ratio (0.0 to 1.0)")
    severity_score: float = Field(..., ge=0.0, le=1.0, description="Fatigue Severity (0.0 saft to 1.0 critical)")
    consecutive_closed_frames: int = Field(...,ge=0, description="Frames eye remained closed")
    alert_type: str = Field(..., pattern="^(DROWSY_WARNING|MICROSLEEP_CRITICAL|YAWN_DETECTED)$")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) 

    @model_validator(mode="after")
    def validate_severity_consistency(self):
        if self.alert_type == "MICROSLEEP_CRITICAL" and self.severity_score < 0.70:
            raise ValueError(
            "MICROSLEEP_CRITICAL alert must carry a severity_score >= 0.70"
        )
        return self

class BatchIngestionResponse(BaseModel):
    status: str
    records_inserted: int
    processed_at: datetime

class IngestionResponse(BaseModel): # IngestionResponse: this defines what our api sends back to the edge device after successfully saving an alert.
    status: str
    record_id: int
    received_at: datetime

# --- API endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK) # @app.get("/health"): creates a web route when someone visist it, it counts how many total logs are in the database and returns a JSON response saying healthy.
def health_check():
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fatigue_incident_logs;")
        total_records = cur.fetchone()[0]
    return {"status": "healthy", "persisted_records": total_records}

@app.post("/api/v1/telemetry/alert", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED) # @app.post(..): creates a route where edge devices can send POST requests with JSON data containing fatigue alerts.
async def ingest_fatigue_alert(payload: FatigueAlertPayload): #FastAPI automatically grabs the incoming JSON, validates it against our FatigueAlertPayload blueprint, and passes it into our function aa payload.
    timestamp_str = payload.timestamp.isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO fatigue_incident_logs (
                driver_id, vehicle_category, ear_score,
                severity_score, consecutive_closed_frames,
                alert_type, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            payload.driver_id,
            payload.vehicle_category,
            payload.ear_score,
            payload.severity_score,
            payload.consecutive_closed_frames,
            payload.alert_type,
            timestamp_str
        ))
        conn.commit()
        inserted_id = cur.lastrowid   # grabs the unique ID number given to this newly created row in the database.
    return IngestionResponse(   # sends a confirmation back to the vehicle device.
        status="persisted_to_db",
        record_id=inserted_id,
        received_at=payload.timestamp
    )

@app.post(
        "/api/v1/telemetry/batch",
        response_model=BatchIngestionResponse,
        status_code=status.HTTP_201_CREATED,
)
async def ingest_fatigue_alerts_batch(
        payloads: List[FatigueAlertPayload]):
    """accepts a micro batch of validated fatigue alerts (upto 500 items)
    inserts all records in a single atomic database transaction via executemany.
    """
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload list cannot be empty."
        )
    if len(payloads) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds max limit of 500 records."
        )
    records_to_insert = [
        (
            item.driver_id,
            item.vehicle_category,
            item.ear_score,
            item.severity_score,
            item.consecutive_closed_frames,
            item.alert_type,
            item.timestamp.isoformat(),
        )
        for item in payloads
    ]

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
                INSERT INTO fatigue_incident_logs (
                    driver_id, vehicle_category, ear_score,
                    severity_score, consecutive_closed_frames,
                    alert_type, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            records_to_insert,
        )
        conn.commit()
    return BatchIngestionResponse(
        status="batch_persisted_to_db",
        records_inserted=len(records_to_insert),
        processed_at=datetime.now(timezone.utc),
    )
   
@app.get("/api/v1/telemetry/alerts/{driver_id}", status_code=status.HTTP_200_OK) # a dynamic route that accepts a driver's ID in the web URL
async def get_driver_alerts(driver_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory=sqlite3.Row # tells sqlite to return rows like dict so we can easily convert them into JSON.
        cur = conn.cursor()
        cur.execute("""
            SELECT incident_id, driver_id, vehicle_category, ear_score,
                    severity_score, consecutive_closed_frames, alert_type, recorded_at
            FROM fatigue_incident_logs
            WHERE driver_id = ?
            ORDER BY recorded_at DESC;
        """, (driver_id,))
        rows = cur.fetchall()
    if not rows:   # if the db finds zero logs for that driver id, it raises a clean 404 error.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No telemetry logs found for driver_id {driver_id}"
        ) 
    events = [dict(row) for row in rows] # converts all retrieved db rows into a clean list of dict and return them alongside the total count.
    return {"driver_id": driver_id, "total_events": len(events), "events": events}

@app.get(
    "/api/v1/analytics/driver-risk-leaderboard", status_code=status.HTTP_200_OK
)
async def get_driver_risk_leaderboard(limit: int = 10):
    """calculates weighted driver safety risk scores directly in sqlite using window functions """
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = """
                WITH DriverAggregates AS (
                    SELECT
                        driver_id,
                        vehicle_category,
                        COUNT(*) AS total_incidents,
                        SUM(CASE WHEN alert_type = 'MICROSLEEP_CRITICAL' THEN 1 ELSE 0 END) AS critical_microsleeps,
                        ROUND(AVG(ear_score), 4) AS avg_ear,
                        ROUND(SUM(
                            severity_score * CASE
                                WHEN alert_type = 'MICROSLEEP_CRITICAL' THEN 3.0
                                WHEN alert_type = 'DROWSY_WARNING' THEN 1.5
                                ELSE 1.0
                            END
                        ), 2) AS composite_risk_score
                    FROM fatigue_incident_logs
                    GROUP BY driver_id
                ),
                RankedDrivers AS (
                    SELECT
                        driver_id,
                        vehicle_category,
                        total_incidents,
                        critical_microsleeps,
                        avg_ear,
                        composite_risk_score,
                        DENSE_RANK() OVER (ORDER BY composite_risk_score DESC) as fleet_risk_rank
                    FROM DriverAggregates
                )
                SELECT * FROM RankedDrivers
                ORDER BY fleet_risk_rank ASC
                LIMIT ?;
            """
        cur.execute(query, (limit,))
        rows = cur.fetchall()
    leaderboard = [dict(row) for row in rows]
    return {"fleet_leaderboard": leaderboard, "total_ranked": len(leaderboard)}
    
if __name__ == "__main__":
    # runs the local development ASGI server
    uvicorn.run("friday_telemetry_api:app", host="127.0.0.1", port=8000, reload=True)