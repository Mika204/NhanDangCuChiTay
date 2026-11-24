
import argparse
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from feature_extraction import extract_features
from preprocessing import ROOT, preprocess_hand
from test_model import load_artifacts

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_mediapipe_model():
    """Kiem tra va tai model hand_landmarker.task neu chua co."""
    model_path = ROOT / "models" / "hand_landmarker.task"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("Dang tai model MediaPipe HandLandmarker (~8MB)...")
        try:
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(model_path))
            print("Tai model MediaPipe thanh cong.")
        except Exception as exc:
            print(f"Khong the tai model MediaPipe: {exc}.")
            return None
    return model_path


def init_mediapipe():
    """Khoi tao MediaPipe HandLandmarker neu kha dung."""
    try:
        import mediapipe as mp
        model_path = ensure_mediapipe_model()
        if model_path is None or not model_path.exists():
            return None

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        return HandLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"MediaPipe khong kha dung ({exc}).")
        return None


def get_mediapipe_bbox(frame, landmarker):
    """Phat hien ban tay bang MediaPipe, tra ve bounding box."""
    import mediapipe as mp

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    box_w = max_x - min_x
    box_h = max_y - min_y

    # Ti le padding ti le chuan (~15% ngang, ~12% doc) theo dung bounding box cua tap HaGRID
    pad_w = box_w * 0.15
    pad_h = box_h * 0.12
    x1 = max(0, int((min_x - pad_w) * w))
    x2 = min(w, int((max_x + pad_w) * w))
    y1 = max(0, int((min_y - pad_h) * h))
    y2 = min(h, int((max_y + pad_h) * h))

    if x2 - x1 < 25 or y2 - y1 < 25:
        return None

    return (x1, y1, x2, y2)


def predict_crop(crop, model, scaler):
    """Trich xuat HOG va du doan cu chi bang model SVM da train."""
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None, None

    stages = preprocess_hand(crop, mask_method="none")
    vector, _ = extract_features(stages, mode="hog")
    scaled = scaler.transform(vector.reshape(1, -1))

    label = str(model.predict(scaled)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)[0]
        confidence = float(probabilities[list(model.classes_).index(label)])

    return label, confidence


class PredictionStabilizer:
    """On dinh ket qua du doan bang Majority Voting qua mot so frame lien tiep."""

    def __init__(self, history_size=7, min_confidence=0.45, vote_ratio=0.55):
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.vote_ratio = vote_ratio
        self.history = []
        self.current_label = None
        self.current_confidence = None

    def update(self, pred_label, pred_conf):
        if pred_label is None or (pred_conf is not None and pred_conf < self.min_confidence):
            self.history.append(("uncertain", pred_conf or 0.0))
        else:
            self.history.append((pred_label, pred_conf or 0.0))

        if len(self.history) > self.history_size:
            self.history.pop(0)

        # Majority voting
        labels = [item[0] for item in self.history]
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]

        if top_label != "uncertain" and (top_count / len(self.history)) >= self.vote_ratio:
            self.current_label = top_label
            confidences = [item[1] for item in self.history if item[0] == top_label]
            self.current_confidence = sum(confidences) / len(confidences) if confidences else pred_conf
        elif top_label == "uncertain":
            self.current_label = None
            self.current_confidence = None

        return self.current_label, self.current_confidence

    def reset(self):
        self.history.clear()
        self.current_label = None
        self.current_confidence = None


def run(camera=0):
    model, scaler, _ = load_artifacts()
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise ValueError(f"Khong mo duoc camera {camera}. Thu --camera 1 hoac kiem tra ket noi.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Khoi tao MediaPipe
    landmarker = init_mediapipe()
    if landmarker is None:
        capture.release()
        raise ValueError("Khong khoi tao duoc MediaPipe HandLandmarker.")

    mirror_mode = True  # Lat guong de trai nghiem truc quan
    active = True       # Tam dung / tiep tuc

    stabilizer = PredictionStabilizer(history_size=7, min_confidence=0.50, vote_ratio=0.55)

    prev_frame_time = time.perf_counter()
    fps = 0.0
    last_predict_time = 0.0
    previous_bbox = None

    print("\n" + "=" * 55)
    print("=== CHUONG TRINH NHAN DIEN CU CHI TAY (HOG + SVM) ===")
    print("=" * 55)
    print("- Che do phat hien: MEDIAPIPE")
    print("- Phim 'Space': Tam dung / Tiep tuc du doan")
    print("- Phim 'm': Bat / Tat lat guong (Mirror)")
    print("- Phim 'q' hoac 'Esc': Thoat")
    print("=" * 55 + "\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Khong doc duoc frame tu webcam.")

            if mirror_mode:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            display = frame.copy()
            now = time.perf_counter()

            # Tinh FPS
            dt = now - prev_frame_time
            prev_frame_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else 1.0 / dt

            bbox = None

            # 1. Xac dinh bounding box theo che do
            bbox = get_mediapipe_bbox(frame, landmarker)
            # 2. Lam muot bounding box giua cac frame de giam rung
            if bbox is not None:
                if previous_bbox is not None:
                    bbox = tuple(
                        round(0.7 * curr + 0.3 * prev)
                        for curr, prev in zip(bbox, previous_bbox)
                    )
                previous_bbox = bbox
                bx1, by1, bx2, by2 = bbox

                # Ve bounding box xanh la bao quanh ban tay
                box_color = (0, 255, 0)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), box_color, 2)

                # 3. Du doan cu chi (moi 0.08s)
                if active and (now - last_predict_time >= 0.08):
                    crop = frame[by1:by2, bx1:bx2]
                    pred_label, pred_conf = predict_crop(crop, model, scaler)
                    stabilizer.update(pred_label, pred_conf)
                    last_predict_time = now
            else:
                previous_bbox = None
                stabilizer.update(None, None)

            # 4. Hien thi giao dien (HUD)
            # Ve thanh tieu de den mo tren dau man hinh
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

            stable_label = stabilizer.current_label
            stable_conf = stabilizer.current_confidence

            if not active:
                status_text = "TAM DUNG (Press Space)"
                status_color = (0, 200, 255)
            elif bbox is None:
                status_text = "Khong phat hien tay"
                status_color = (0, 0, 255)
            elif stable_label:
                conf_str = f" ({stable_conf:.1%})" if stable_conf else ""
                status_text = f"CU CHI: {stable_label.upper()}{conf_str}"
                status_color = (0, 255, 0)
            else:
                status_text = "DANG NHAN DIEN..."
                status_color = (255, 255, 255)

            cv2.putText(display, status_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

            mode_str = f"Mode: MEDIAPIPE | Mirror: {'ON' if mirror_mode else 'OFF'} | FPS: {int(round(fps))}"
            cv2.putText(display, mode_str, (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            # Huong dan phim tat ben duoi
            help_text = "Space: Pause | M: Mirror | Q: Quit"
            cv2.putText(display, help_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Gesture Webcam Demo", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Gesture Webcam Demo", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                active = not active
                if not active:
                    stabilizer.reset()
            elif key == ord("m"):
                mirror_mode = not mirror_mode
                stabilizer.reset()
                print(f"-> Lat guong (Mirror): {'ON' if mirror_mode else 'OFF'}")

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam demo nhan dien cu chi tay (HOG + SVM)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (mac dinh: 0)")
    args = parser.parse_args()
    try:
        run(args.camera)
    except (OSError, ValueError, cv2.error) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
