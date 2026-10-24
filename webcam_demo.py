"""Buoc 10: webcam voi ROI thu cong, HOG + model da luu."""

import argparse
import time

import cv2

from feature_extraction import extract_features
from preprocessing import preprocess_hand
from test_model import load_artifacts


def predict_crop(crop, model, scaler):
    stages = preprocess_hand(crop, mask_method="none")
    vector, _ = extract_features(stages, mode="hog")
    scaled = scaler.transform(vector.reshape(1, -1))
    label = str(model.predict(scaled)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(scaled)[0, list(model.classes_).index(label)])
    return label, confidence, stages["clahe"]


def run(camera=0):
    model, scaler, _ = load_artifacts()
    capture = cv2.VideoCapture(camera)
    try:
        if not capture.isOpened():
            raise ValueError(f"Khong mo duoc camera {camera}. Thu --camera 1 hoac dong ung dung dang dung camera.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        roi = None
        active = False
        label, confidence = "Paused", None
        last_prediction = 0.0
        print("R: chon vung tay (Enter xac nhan va tu bat du doan, C huy).")
        print("Space: bat/tat du doan. Q/Esc: thoat.")
        print("Khoanh sat ban tay; giu tay trong khung. Khong tu dong phat hien/theo doi tay.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Khong doc duoc frame tu webcam.")
            # Giu huong anh goc; khong lat ngang dau vao so voi pipeline train.
            height, width = frame.shape[:2]
            if roi is None:
                side = min(width, height) // 2
                roi = ((width - side) // 2, (height - side) // 2, side, side)
            x, y, w, h = roi
            if x < 0 or y < 0 or x + w > width or y + h > height:
                roi = None
                active = False
                continue
            if active and time.perf_counter() - last_prediction >= 0.2:
                label, confidence, preview = predict_crop(frame[y:y + h, x:x + w], model, scaler)
                last_prediction = time.perf_counter()
            display = frame.copy()
            cv2.rectangle(display, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            text = f"Gesture: {label}" if active else "Paused - put hand in ROI, press Space"
            cv2.putText(display, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if active and confidence is not None:
                cv2.putText(display, f"Confidence: {confidence:.1%}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "R: select ROI | Space: predict/pause | Q: quit", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Gesture webcam", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Gesture webcam", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                active = False
                selected = cv2.selectROI("Select hand - Enter confirm", frame,
                                        showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("Select hand - Enter confirm")
                if selected[2] >= 16 and selected[3] >= 16:
                    roi = tuple(int(v) for v in selected)
                    active = True
                    last_prediction = 0.0
                    print("ROI:", roi, "- dang du doan; xem Gesture va Confidence tren cua so webcam.")
                else:
                    print("ROI huy hoac qua nho; giu ROI truoc.")
                label, confidence = "Waiting" if active else "Paused", None
            if key == ord(" "):
                active = not active
                label, confidence = "Waiting", None
                last_prediction = 0.0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo HOG voi ROI thu cong")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args.camera)
    except (OSError, ValueError, cv2.error) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
