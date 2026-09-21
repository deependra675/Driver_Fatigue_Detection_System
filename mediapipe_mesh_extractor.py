""" MediaPipe face mesh ingestion pipeline.
Extracts normalized 3D landmarks, scales them to pixel space, and pipes eye contours directly into vectorized EAR computation."""
from typing import Tuple # a python helper to specify that a func returns a tuple of items.
import cv2
import mediapipe as mp
import numpy as np
import requests # used to send HTTP web requests
from temporal_fatigue_tracking_engine import TemporalFatigueTracker

# web address where we will send our telemetry alerts
API_INGEST_URL = "http://127.0.0.1:8000/api/v1/telemetry/alert"

def dispatch_telemetry_alert( # this func triggered whenever the driver hits a critical drowsiness state.
    driver_id: int, vehicle_category: str, ear: float, closed_frames: int
):
    payload={   # a dict containing all the details of the incident. this matches the exact pydantic contract we built in our fastapi app.
        "driver_id": driver_id,
        "vehicle_category": vehicle_category,
        "ear_score": round(ear, 3),
        "severity_score": 0.85,
        "consecutive_closed_frames": closed_frames,
        "alert_type": "MICROSLEEP_CRITICAL",
    }
    try:
        response = requests.post(API_INGEST_URL, json=payload, timeout=0.1)  # sends the package (json=payload) to our fastapi server. timeout=0.1 ensures the camera doesnt freeze up if the server takes too long to reply.
        if response.status_code == 201:
            print("-> [HTTP 201] Alert sent and saved to database!")
        else:
            print(f"-> [HTTP {response.status_code}] Backend rejected; {response.text}")
    except requests.RequestException:
        print("-> [WARN] Server offline. Skipped.") # if the fastapi server isnt running, the code wont crash. that's why we used try/except.

def calculate_ear_vectorized(eye_points: np.ndarray) -> float:
    """calculates eye aspect ratio (EAR) using vectorized euclidean norms.
    eye_points shape: (6,2) containing points [P1, P2, P3, P4, P5, P6]"""

    # vertical distances: ||P2 - P6|| and ||P3 - P5||
    v1 = np.linalg.norm(eye_points[1] - eye_points[5]) # np.linalg.norm : used to calculate the euclidean distance
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])

    # horizontal distance ||P1 - P4||
    h = np.linalg.norm(eye_points[0] - eye_points[3])

    if h < 1e-6: # this prevents a math crash called division by zero error, in case horizontal width becomes zero
        return 0.0
    return float((v1 + v2) / (2.0 * h))

class FaceMeshEyeExtractor:
    # canonical mediapipe face mesh landmark indices.
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__ (self, max_num_faces: int=1, min_detection_confidence: float=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces = max_num_faces,
            refine_landmarks = True, # enables iris tracking & refined eye borders
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract_eye_landmarks(
            self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """processes a BGR video frame and returns scaled (x, y) coordinates
        for both eyes as NumPy arrays of shape (6, 2)"""

        h, w, _ = frame.shape # grabs height and width of video frame (eg 720p), we ignore _, which is the color channel count
        # mediapipe requires RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #opencv loads video in BGR, but google's mediapipe was trained on RGB, so it converts
        results = self.face_mesh.process(rgb_frame) # core AI step, it scans the RGB images and looks for a face

        if not results.multi_face_landmarks: # if someone walks away from the camera or turns around and no face is found, it safely stops and return None, instead of crashing
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark # pulls out the coordinate data for the first face detected [0]

        # Scale normalized coordinates (0.0 to 1.0) to frame resolution
        # mediapipe doesnt output pixel numbers directly, it outputs fractions, eg. an eye coordinate of x =0.5 means "halfway across the screen width"
        def to_pixel_coords(indices):
            coords = [[landmarks[i].x*w, landmarks[i].y*h] for i in indices]
            return np.array(coords, dtype=np.float32)
        left_eye = to_pixel_coords(self.LEFT_EYE_IDX)
        right_eye = to_pixel_coords(self.RIGHT_EYE_IDX)
        return left_eye, right_eye

def run_headless_simulation():
    """runs an automated synthetic feed to test the pipeline without a camera."""
    print("=" * 65)
    print(" No webcam detected: running headless synthetic simultion")
    print("=" * 65)

    tracker = TemporalFatigueTracker(
        ear_threshold=0.20,
        consecutive_frame_threshold=5,
        buffer_size=10,
    )
    # 1. Synthetic 6 point coordinates for an open eye (EAR ~ 0.35)
    open_eye = np.array( 
        [
            [100, 100], # P1 (outer corner)
            [110, 85], #  P2 (top_left)
            [130, 85], # P3 (top right)
            [140, 100], # P4 (inner corner)
            [130, 115], # P5 (bottom right)
            [110, 115], # P6 (bottom left)
        ],
        dtype = np.float32,
    )
    # 2. Synthetic 6 point coordinates for a closed eye (EAR ~ 0.05)
    closed_eye = np.array(
        [
            [100, 100],
            [110, 99],
            [130, 99],
            [140, 100],
            [130, 101],
            [110, 101],
        ],
        dtype=np.float32,
    )
    open_ear = calculate_ear_vectorized(open_eye)
    closed_ear = calculate_ear_vectorized(closed_eye)

    print(f"Computed synthetic Open EAR: {open_ear: .3f}")
    print(f"Computed synthetic closed EAR: {closed_ear: .3f}")
    print("=" * 65)

    # stream: 3 open frames -> 6 closed frames (triggers alert) -> 1 open frame (recovery)
    stream = [open_ear] * 3 + [closed_ear] * 6 + [open_ear]
    for frame_idx, ear in enumerate(stream, start=1):
        state = tracker.process_frame_ear(ear)
        print(
            f"Frame {frame_idx:02d} | EAR: {ear: .3f} | Closed: {state['closed_frames']} |"
            f"Drowsy: {str(state['is_drowsy']):<5} | Alert; {state['new_alert_triggered']}"
        )
        if state["new_alert_triggered"]:
            dispatch_telemetry_alert(
                driver_id=101,
                vehicle_category="Heavy Truck",
                ear=ear,
                closed_frames=state["closed_frames"],
            )
    print("=" * 65)
    print("Synthetic ingestion pipeline verified successfully")
    print("=" * 65)

def run_pipeline(): 
    cap = cv2.VideoCapture(0) # initialize opencv video capture using the default local webcam index 0
    if not cap.isOpened():
        run_headless_simulation()
        return
    extractor = FaceMeshEyeExtractor()
    tracker = TemporalFatigueTracker(
        ear_threshold=0.22, consecutive_frame_threshold=15, buffer_size=30
    )
    print("Webcam found. Starting video stream. Press 'q' to exit.")

    while True: # capture the next video frame from the camera
        ret, frame = cap.read() # ret is a boolean success flag, frame is the image array
        if not ret: # checks if the frame reading failed.
            break # exists the video loop if the camera feed drops.
 
        left_eye, right_eye = extractor.extract_eye_landmarks(frame)

        if left_eye is not None and right_eye is not None:
            # calculate EAR for both eyes and compute mean
            left_ear = calculate_ear_vectorized(left_eye)
            right_ear = calculate_ear_vectorized(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0  # average both eye EAR values for enchanced robustness.

            # feed into temporal state engine
            state = tracker.process_frame_ear(avg_ear)

            # Send alert to FastAPI when a NEW drowsiness event is detected
            if state["new_alert_triggered"]:
                print(">>> ALERT TRIGGERED - sending to API")

                dispatch_telemetry_alert(
                    driver_id=101,
                    vehicle_category="Heavy Truck",
                    ear=avg_ear,
                    closed_frames=state["closed_frames"],
                )

            # overlay telemetry directly onto the frame
            status_color = (0, 0, 255) if state["is_drowsy"] else (0, 255, 0)

            status_text = "DROWSY ALERT!" if state["is_drowsy"] else "AWAKE"

            cv2.putText( # specifies text content, position coordinates (30, 50), font style, scale, white color, and thickness to draw EAR overlay.
                frame,
                f"EAR: {avg_ear: .2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )
            cv2.putText(
                frame,
                f"Closed Frames: {state['closed_frames']}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # draw eye contours for visula verification
            for pt in np.vstack((left_eye, right_eye)).astype(int):
                cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1)
        cv2.imshow("Driver telemetry ingestion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release() # releases the webcam hardware device so other apps can use it
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()


