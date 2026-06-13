import cv2
import supervision as sv
from ultralytics import YOLO
from collections import Counter
import torch
import threading

class ObjectDetector:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 Using Apple Silicon MPS GPU!")
        else:
            self.device = "cpu"
            print("⚠️ Using CPU")

        # ONE medium model — best balance of speed & accuracy
        self.model = YOLO("yolo11m.pt")
        self.model.to(self.device)

        # Thread lock — prevents simultaneous inference crashes
        self.model_lock = threading.Lock()

        # self.CONFIDENCE = 0.30
        # self.FRAME_SKIP = 3
        self.CONFIDENCE = 0.50  # Higher confidence for fewer false positives
        self.FRAME_SKIP = 5     # Process every 5th frame for better performance

        # Per camera — all lightweight
        self.frame_counters = {}
        self.trackers = {}
        self.last_detections = {}
        self.last_labels = {}

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def process_frame(self, frame, cam_id):
        if cam_id not in self.frame_counters:
            self.frame_counters[cam_id] = 0
            self.trackers[cam_id] = sv.ByteTrack()
            self.last_detections[cam_id] = None
            self.last_labels[cam_id] = []

        self.frame_counters[cam_id] += 1

        if self.frame_counters[cam_id] % self.FRAME_SKIP == 0:

            # Locked inference — one thread at a time
            with self.model_lock:
                results = self.model(
                    frame,
                    conf=self.CONFIDENCE,
                    verbose=False
                )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = self.trackers[cam_id].update_with_detections(
                detections
            )

            labels = []
            if len(detections) > 0:
                for tracker_id, class_id in zip(
                    detections.tracker_id,
                    detections.class_id
                ):
                    labels.append(
                        f"#{tracker_id} {self.model.names[class_id]}"
                    )

            self.last_detections[cam_id] = detections
            self.last_labels[cam_id] = labels

        detections = self.last_detections[cam_id]
        labels = self.last_labels[cam_id]

        annotated = frame.copy()
        if detections is not None and len(detections) > 0:
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(
                annotated, detections, labels
            )

        return annotated, detections, labels

    def get_scene_summary(self, detections):
        if detections is None or len(detections) == 0:
            return "No activity detected."

        objects = [
            self.model.names[class_id]
            for class_id in detections.class_id
        ]
        counts = Counter(objects)
        return ", ".join(
            f"{count} {obj}"
            for obj, count in counts.items()
        )