from ultralytics import YOLO


class DemoDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.4):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def classify_label(self, class_name):
        name = class_name.lower()

        recycling_keywords = [
            "bottle", "cup", "can", "glass", "paper", "cardboard", "carton"
        ]
        trash_keywords = [
            "banana", "apple", "orange", "food", "chip", "snack", "wrapper",
            "styrofoam", "foam"
        ]

        if any(word in name for word in recycling_keywords):
            return "Recycling"
        if any(word in name for word in trash_keywords):
            return "Trash"
        return "Unknown"

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                class_name = names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "category": self.classify_label(class_name),
                    "confidence": confidence
                })

        return detections