import cv2


def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Could not open camera index {index}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Opened camera index {index}")
    return cap


def read_frame(cap):
    if cap is None:
        return None

    ok, frame = cap.read()
    if not ok:
        return None

    return frame


def close_camera(cap):
    if cap is not None:
        cap.release()