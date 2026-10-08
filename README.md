# Edge Driver Fatigue & Fleet Telemetry Engine

An asynchronous, edge-to-cloud driver fatigue monitoring and real-time telemetry service. Built with MediaPipe Face Mesh, temporal sliding-window state tracking, and FastAPI with SQLite WAL persistence.

## System Architecture

```text
       [ Webcam / Dashcam Feed ]
                   │
                   ▼
       [ MediaPipe Face Mesh ]
     (468/478 Landmark Extraction)
                   │
                   ▼
     [ Vectorized EAR Computation ]
        (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
                   │
                   ▼
     [ Temporal Fatigue Tracker ]
    (O(1) Bounded Ring Buffer Deque)
    (Rising-Edge Alert Transition)
                   │
         Alert Triggered?
        ┌──────────┴──────────┐
       No                    Yes
        │                     │
     Continue        [ Edge Dispatcher ]
                     (HTTP 201 / Offline Disk Spooling)
                              │
                              ▼
                   [ FastAPI Telemetry API ]
                   (Pydantic Validation & SQLite WAL)
                              │
                              ▼
                [ Rolling Time-Series Analytics ]
             (15-min Window Burst & Risk Profiling)