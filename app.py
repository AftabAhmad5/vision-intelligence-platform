from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from detector import ObjectDetector
from llm import SceneAnalyzer

app = Flask(__name__)

# ONE shared detector and analyzer
detector = ObjectDetector()
analyzer = SceneAnalyzer()

# Video sources
CAMERAS = {
    "cam1": "cam1.mp4",
    "cam2": "cam2.mp4",
    "cam3": "cam3.mp4",
    "cam4": "cam4.mp4"
}

# Store latest data per camera
scene_summaries = {
    "cam1": "Waiting...",
    "cam2": "Waiting...",
    "cam3": "Waiting...",
    "cam4": "Waiting..."
}

llava_descriptions = {
    "cam1": "Initializing...",
    "cam2": "Initializing...",
    "cam3": "Initializing...",
    "cam4": "Initializing..."
}

summary_lock = threading.Lock()

def generate_frames(cam_id):
    cap = cv2.VideoCapture(CAMERAS[cam_id])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for performance
        frame = cv2.resize(frame, (640, 360))

        # YOLO detection
        annotated_frame, detections, labels = detector.process_frame(
            frame, cam_id
        )

        # Get YOLO summary
        yolo_summary = detector.get_scene_summary(detections)

        # LLaVA scene analysis
        llava_desc = analyzer.analyze(frame, cam_id, yolo_summary)

        # Update summaries thread safely
        with summary_lock:
            scene_summaries[cam_id] = yolo_summary
            llava_descriptions[cam_id] = llava_desc

        # Encode frame
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<cam_id>')
def video_feed(cam_id):
    return Response(
        generate_frames(cam_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/summaries')
def get_summaries():
    with summary_lock:
        return jsonify({
            "yolo": scene_summaries,
            "llava": llava_descriptions
        })

if __name__ == '__main__':
    app.run(debug=False, threaded=True)