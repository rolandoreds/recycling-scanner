import cv2

from camera import open_camera, read_frame, close_camera
from inference import DemoDetector
from label_map import FINE_TO_PUBLIC_LABEL


WINDOW_NAME = "Recycling Scanner"
CAMERA_INDEX = 0


def get_label(det):
    return (
        det.get("label")
        or det.get("class_name")
        or det.get("name")
        or str(det.get("class", "Unknown"))
    )


def get_confidence(det):
    value = det.get("confidence", det.get("score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_box(det):
    box = det.get("box", det.get("bbox"))

    if box is None:
        return None

    if len(box) != 4:
        return None

    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2), int(y2)


def draw_ui(frame, detections):
    overlay = frame.copy()

    # Title
    cv2.putText(
        overlay,
        "Recycling Scanner",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # Status block
    cv2.rectangle(overlay, (20, 60), (520, 200), (30, 30, 30), -1)
    cv2.putText(
        overlay,
        "Detected Items:",
        (35, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if not detections:
        cv2.putText(
            overlay,
            "No items detected",
            (35, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )
    else:
        y_text = 130
        for det in detections[:3]:
            fine_label = get_label(det)
            public_label = FINE_TO_PUBLIC_LABEL.get(fine_label, "Other")
            confidence = get_confidence(det)

            text = f"{fine_label} -> {public_label} ({confidence:.2f})"
            cv2.putText(
                overlay,
                text,
                (35, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_text += 30

    # Draw boxes
    for det in detections:
        box = get_box(det)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        fine_label = get_label(det)
        public_label = FINE_TO_PUBLIC_LABEL.get(fine_label, "Other")
        confidence = get_confidence(det)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            overlay,
            f"{fine_label} | {public_label} | {confidence:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return overlay


def main():
    cap = open_camera(CAMERA_INDEX, 640, 480)
    detector = DemoDetector()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("Failed to read frame.")
            continue

        detections = detector.detect(frame)

        print("detections:", detections)

        output = draw_ui(frame, detections)
        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

    close_camera(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()