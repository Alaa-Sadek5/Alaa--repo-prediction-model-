import cv2
import threading
import numpy as np
import joblib
from collections import deque
from ultralytics import YOLO

class TrafficAI:
    def __init__(self, model_path="yolo11n.pt", video_path="my_traffic.mp4", lag_steps=5):
        self.model = YOLO(model_path)
        self.current_video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.target_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
        self.lock = threading.Lock()
        
        self.latest_count = 0
        self.latest_density = "0%"
        self.latest_level = "UNKNOWN"
        self.latest_jpeg = None

        # XGBoost Prediction Setup
        self.lag_steps = lag_steps
        self.count_history = deque(maxlen=lag_steps)
        self.predicted_count = 0
        self.predicted_level = "UNKNOWN"

        # Load trained XGBoost model
        try:
            self.xgb_model = joblib.load("traffic_xgb_model.pkl")
            print("[INFO] Loaded XGBoost traffic prediction model successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load XGBoost model: {e}")
            self.xgb_model = None

    def set_video(self, new_video_path):
        with self.lock:
            if self.current_video_path != new_video_path:
                self.current_video_path = new_video_path
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(new_video_path)
                self.count_history.clear()

    def _predict_xgboost(self):
        """Predicts future vehicle count using trained XGBoost model and lag features."""
        if self.xgb_model is None or len(self.count_history) < self.lag_steps:
            return self.latest_count, self.latest_level

        # Convert recent count history into lag feature array [lag_5, lag_4, lag_3, lag_2, lag_1]
        features = np.array(list(self.count_history)).reshape(1, -1)
        
        # Predict count with XGBoost model
        pred_val = self.xgb_model.predict(features)[0]
        pred_count = max(0, int(round(pred_val)))

        # Determine predicted congestion level
        if pred_count < 10:
            pred_level = "LOW"
        elif pred_count <= 25:
            pred_level = "MODERATE"
        else:
            pred_level = "HIGH"

        return pred_count, pred_level

    def update_frame(self):
        """Read frame, detect vehicles with YOLO, predict with XGBoost, and render overlays."""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return False

            # Run YOLO detection
            results = self.model(frame, classes=self.target_classes, verbose=False)
            count = len(results[0].boxes)

            # Record count into history buffer for lag features
            self.count_history.append(count)

            # Determine current congestion level
            if count < 10:
                level = "LOW"
            elif count <= 25:
                level = "MODERATE"
            else:
                level = "HIGH"

            # Calculate traffic density
            max_capacity = 30
            density_val = min(100, int((count / max_capacity) * 100))

            # Run XGBoost inference
            self.predicted_count, self.predicted_level = self._predict_xgboost()

            # Save latest metrics
            self.latest_count = count
            self.latest_density = f"{density_val}%"
            self.latest_level = level

            # Annotate video frame
            annotated_frame = results[0].plot()

            # Display prediction overlay on video stream
            overlay_text = f"XGB Pred: {self.predicted_count} vehicles ({self.predicted_level})"
            cv2.putText(annotated_frame, overlay_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            self.latest_jpeg = buffer.tobytes()
            return True

    def get_metrics(self):
        return {
            "vehicle_count": self.latest_count,
            "traffic_density": self.latest_density,
            "congestion_level": self.latest_level,
            "predicted_vehicle_count": self.predicted_count,
            "predicted_congestion_level": self.predicted_level
        }

    def get_jpeg_frame(self):
        return self.latest_jpeg
