import requests
import base64
import cv2
import threading


class SceneAnalyzer:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "llava"
        # self.ANALYZE_EVERY = 30
        self.ANALYZE_EVERY = 60  # Analyze every 60 frames for better performance   
        self.frame_counters = {}
        self.last_descriptions = {}
        self.analysis_threads = {}
        self.thread_lock = threading.Lock()
        print("🧠 LLaVA Scene Analyzer ready!")

    def _frame_to_base64(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, frame, cam_id, yolo_summary):
        if cam_id not in self.frame_counters:
            self.frame_counters[cam_id] = 0
            self.last_descriptions[cam_id] = "Initializing scene analysis..."
            self.analysis_threads[cam_id] = False

        self.frame_counters[cam_id] += 1

        if (self.frame_counters[cam_id] % self.ANALYZE_EVERY == 0 and
                not self.analysis_threads.get(cam_id, False)):

            thread = threading.Thread(
                target=self._analyze_background,
                args=(frame.copy(), cam_id, yolo_summary),
                daemon=True
            )
            self.analysis_threads[cam_id] = True
            thread.start()

        return self.last_descriptions[cam_id]

    def _analyze_background(self, frame, cam_id, yolo_summary):
        try:
            image_b64 = self._frame_to_base64(frame)

            prompt = f"""You are a surveillance camera analyst.
Look at this camera feed and describe what is happening in 2 sentences maximum.
YOLO already detected: {yolo_summary}
Focus on behavior, movement, and any unusual activity.
Be concise and professional."""

            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False
                },
                timeout=15
            )

            if response.status_code == 200:
                description = response.json().get("response", "").strip()
                with self.thread_lock:
                    self.last_descriptions[cam_id] = description

        except Exception as e:
            print(f"LLaVA error on {cam_id}: {e}")
        finally:
            self.analysis_threads[cam_id] = False