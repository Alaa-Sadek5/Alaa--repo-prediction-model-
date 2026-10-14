import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from traffic_v11_ai import TrafficAI

app = FastAPI(title="Smart Traffic Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

traffic_detector = TrafficAI(model_path="yolo11n.pt", video_path="videos/video1.mp4")

class VideoRequest(BaseModel):
    video_name: str

@app.get("/")
def root():
    return {"status": "Traffic API is running"}

@app.get("/traffic-data")
def get_traffic_data():
    return traffic_detector.get_metrics()

@app.post("/change-video")
def change_video(req: VideoRequest):
    traffic_detector.set_video(req.video_name)
    return {"message": f"Switched to {req.video_name}"}

def generate_mjpeg():
    while True:
        success = traffic_detector.update_frame()
        if not success:
            time.sleep(0.1)
            continue
        frame_bytes = traffic_detector.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")
