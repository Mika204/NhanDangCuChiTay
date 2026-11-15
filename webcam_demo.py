import argparse
import time

import cv2
import numpy as np

from feature_extraction import extract_features
from preprocessing import preprocess_hand
from test_model import load_artifacts


def segment_skin(frame):
    """Phan doan vung da chinh xac bang cach ket hop ca YCrCb va HSV de tranh bat nham nen vang/go/bong toi."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 1. YCrCb: Tach kenh do sang va kenh mau, dat nguong Y >= 70 de loai bo bong toi va toc den
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, (70, 133, 77), (255, 173, 127))

    # 2. HSV: Them ca vung Hue do wrap-around de bat duoc da bi anh sang lam do
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_hsv_warm = cv2.inRange(hsv, (0, 30, 60), (25, 200, 255))
    mask_hsv_red = cv2.inRange(hsv, (160, 20, 60), (180, 200, 255))
    mask_hsv = cv2.bitwise_or(mask_hsv_warm, mask_hsv_red)

    # Giao hai mask de co vung mau da that su tin cay
    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # 3. Phep toan hinh thai hoc: Opening khu nhieu hat, Closing lam lien khoi ban tay
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def isolate_hand_from_arm(cnt, frame_w, frame_h, mask=None):
    """Chi crop dung phan ban tay (tu dau ngon tay den co tay), cat bo phan cang tay keo dai."""
    x, y, w, h = cv2.boundingRect(cnt)
    original_w = w

    # Ty le chieu dai / chieu rong cua ban tay nguoi tu co tay den dau ngon tay thuong la 1.0 - 1.25.
    # Neu chieu cao h vuot qua 1.25 * w, phan du ra ben duoi la cang tay/canh tay.
    arm_from_bottom = (y + h) >= (frame_h - 40) or (h > 1.25 * w)
    arm_from_right = (x + w) >= (frame_w - 30) and (w > 1.25 * h)
    arm_from_left = (x <= 30) and (w > 1.25 * h)
    used_wrist_cut = False

    if arm_from_bottom:
        wrist_cut = None
        top_density = None
        if mask is not None:
            profile = np.count_nonzero(mask[y:y + h, x:x + w], axis=1)
            if profile.size and np.max(profile) > 0:
                top_density = profile[0] / np.max(profile)
            start = int(w * 0.4)
            end = min(h, int(w * 1.7))
            if end - start >= 7:
                smoothed = np.convolve(profile, np.ones(9) / 9, mode="same")
                last_peak = None
                for row in range(start + 1, end - 1):
                    if smoothed[row] > smoothed[row - 1] and smoothed[row] >= smoothed[row + 1]:
                        last_peak = row
                    elif (
                        smoothed[row] < smoothed[row - 1]
                        and smoothed[row] <= smoothed[row + 1]
                        and last_peak is not None
                        and smoothed[row] < smoothed[last_peak] * 0.8
                    ):
                        wrist_cut = row

        if wrist_cut is not None:
            hand_region = mask[y:y + wrist_cut, x:x + w]
            ys, xs = np.where(hand_region > 0)
            if xs.size:
                x += int(xs.min())
                w = int(xs.max() - xs.min() + 1)
                h = max(20, wrist_cut)
                used_wrist_cut = True
            else:
                wrist_cut = None
        if wrist_cut is None:
            h = max(20, min(h, int(w * 1.25)))
    elif arm_from_right:
        # Canh tay dua tu ben phai vao: ban tay o ben trai
        hand_w = min(w, int(h * 1.25))
        w = max(20, hand_w)
    elif arm_from_left:
        # Canh tay dua tu ben trai vao: ban tay o ben phai
        hand_w = min(w, int(h * 1.25))
        x = x + w - hand_w
        w = max(20, hand_w)

    # Them padding nho (khoang 7%) quanh ban tay de khong bi cat sat ngon tay
    pad_x = max(int(w * 0.07), int(original_w * 0.08))
    pad_y = int(h * (0.03 if arm_from_bottom and used_wrist_cut else 0.07))
    pad_top = pad_y
    if used_wrist_cut and top_density is not None and 0.03 <= top_density <= 0.2:
        pad_top = max(pad_top, int(original_w * 0.22))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y)

    return (x1, y1, x2, y2)


def score_hand_candidate(cnt, frame_w, frame_h):
    """Cham diem xem contour co phai la ban tay hay khong (phan biet voi mat hoac than nguoi)."""
    area = cv2.contourArea(cnt)
    if area < 2500 or area > (frame_w * frame_h * 0.65):
        return -1.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return -1.0

    solidity = area / hull_area
    x, y, w, h = cv2.boundingRect(cnt)
    cx, cy = x + w / 2.0, y + h / 2.0

    # Kiem tra xem vung co giong khuon mat khong:
    # Khuon mat thuong o nua tren man hinh, rat dac (solidity > 0.88), ti le w/h gan 0.7-1.0
    is_face_like = (cy < frame_h * 0.5) and (solidity > 0.88) and ((y + h) < frame_h * 0.7)

    # Dem do loi lom (khe giua cac ngon tay)
    defect_score = 0.0
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if len(hull_indices) > 3 and len(cnt) > 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is not None:
                for item in defects:
                    depth = item.ravel()[3] / 256.0
                    if depth > 15.0:
                        defect_score += min(depth, 50.0)
        except cv2.error:
            pass

    # Neu qua giong khuon mat va khong co dau hieu ngon tay, giam manh diem so
    if is_face_like and defect_score < 10.0:
        return area * 0.05

    # Ban tay thuong co khe ngon tay hoac vung dua len tu duoi
    score = area + defect_score * 150.0
    if (y + h) >= frame_h * 0.55:
        score *= 1.3

    return score


def is_neck_candidate(cnt, face_boxes):
    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0:
        return False

    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area else 0.0
    center_x = x + w / 2
    aspect_ratio = w / h

    for face_x1, face_y1, face_x2, face_y2 in face_boxes:
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        face_center_x = (face_x1 + face_x2) / 2
        if (
            face_y2 <= y <= face_y2 + face_height * 0.35
            and abs(center_x - face_center_x) <= face_width * 0.3
            and 0.75 <= aspect_ratio <= 1.25
            and solidity >= 0.9
        ):
            return True
    return False


def extend_bbox_toward_face(bbox, face_boxes, frame_w):
    x1, y1, x2, y2 = bbox
    for face_x1, face_y1, face_x2, face_y2 in face_boxes:
        face_width = face_x2 - face_x1
        if y1 >= face_y2 or y2 <= face_y1:
            continue
        if 0 <= x1 - face_x2 <= face_width * 0.15:
            x1 = min(x1, face_x2 - int(face_width * 0.35))
        elif 0 <= face_x1 - x2 <= face_width * 0.15:
            x2 = max(x2, face_x1 + int(face_width * 0.35))
    return max(0, x1), y1, min(frame_w, x2), y2


def detect_hand_bbox(frame, face_cascade, previous_bbox=None):
    """Tim contour ban tay phu hop nhat va tra ve bounding box chi bao quanh ban tay."""
    height, width = frame.shape[:2]
    mask = segment_skin(frame)

    small_width, small_height = max(1, width // 2), max(1, height // 2)
    small_frame = cv2.resize(
        frame, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    scale_x, scale_y = width / small_width, height / small_height
    face_boxes = []
    for face_x, face_y, face_w, face_h in faces:
        x1, y1 = int(face_x * scale_x), int(face_y * scale_y)
        x2 = int((face_x + face_w) * scale_x)
        y2 = int((face_y + face_h) * scale_y)
        face_boxes.append((x1, y1, x2, y2))
        pad_x, pad_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        mask[y1:y2, x1:x2] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    candidates = []
    for cnt in contours:
        if is_neck_candidate(cnt, face_boxes):
            continue
        score = score_hand_candidate(cnt, width, height)
        if score < 2000.0:
            continue
        bbox = isolate_hand_from_arm(cnt, width, height, mask)
        bbox = extend_bbox_toward_face(bbox, face_boxes, width)
        x1, y1, x2, y2 = bbox
        if (x2 - x1) >= 25 and (y2 - y1) >= 25:
            candidates.append((score, bbox))

    if not candidates:
        return None, mask

    _, bbox = max(candidates, key=lambda item: item[0])
    if previous_bbox is not None:
        previous_cx = (previous_bbox[0] + previous_bbox[2]) / 2
        previous_cy = (previous_bbox[1] + previous_bbox[3]) / 2
        max_distance = float(np.hypot(width, height)) * 0.3
        nearby = []
        for score, candidate_bbox in candidates:
            center_x = (candidate_bbox[0] + candidate_bbox[2]) / 2
            center_y = (candidate_bbox[1] + candidate_bbox[3]) / 2
            distance = float(np.hypot(center_x - previous_cx, center_y - previous_cy))
            if distance <= max_distance:
                nearby.append((distance, score, candidate_bbox))
        if nearby:
            _, _, bbox = min(nearby, key=lambda item: item[0])

        bbox = tuple(
            round(0.65 * current + 0.35 * previous)
            for current, previous in zip(bbox, previous_bbox)
        )

    return bbox, mask


def predict_crop(crop, model, scaler):
    """Dua crop ban tay vao dung pipeline preprocessing va HOG khi train model."""
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None, None, None

    stages = preprocess_hand(crop, mask_method="none")
    vector, _ = extract_features(stages, mode="hog")
    scaled = scaler.transform(vector.reshape(1, -1))

    label = str(model.predict(scaled)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)[0]
        confidence = float(probabilities[list(model.classes_).index(label)])

    return label, confidence, stages["clahe"]


def stabilize_prediction(label, confidence, history, pred_label, pred_conf):
    if label is None:
        return pred_label, pred_conf, [(pred_label, pred_conf)]

    history = (history + [(pred_label, pred_conf)])[-3:]
    if pred_label == label:
        if pred_conf is not None:
            confidence = (
                pred_conf if confidence is None else 0.75 * confidence + 0.25 * pred_conf
            )
    elif sum(item[0] == pred_label for item in history) >= 2:
        label = pred_label
        matching_confidences = [
            item[1] for item in history if item[0] == label and item[1] is not None
        ]
        confidence = (
            sum(matching_confidences) / len(matching_confidences)
            if matching_confidences
            else pred_conf
        )
    return label, confidence, history


def run(camera=0):
    model, scaler, _ = load_artifacts()
    capture = cv2.VideoCapture(camera)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        capture.release()
        raise OSError("Khong tai duoc bo nhan dien khuon mat cua OpenCV.")

    try:
        if not capture.isOpened():
            raise ValueError(
                f"Khong mo duoc camera {camera}. Thu --camera 1 hoac kiem tra ket noi webcam."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=== CHUONG TRINH NHAN DIEN CU CHI TAY TU DONG ===")
        print("- He thong tu dong phan doan da tay, loai bo mat va cat dung ban tay.")
        print("- Phim 'q' hoac 'Esc': Thoat.")

        show_mask_window = False
        label = None
        confidence = None
        prediction_history = []
        last_prediction_time = 0.0
        previous_bbox = None
        missed_frames = 0

        prev_frame_time = time.perf_counter()
        fps = 0.0

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Khong doc duoc frame tu webcam.")

            now = time.perf_counter()
            dt = now - prev_frame_time
            prev_frame_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps = current_fps if fps == 0.0 else (0.85 * fps + 0.15 * current_fps)

            height, width = frame.shape[:2]
            display = frame.copy()

            # 1. Tu dong tim vung ban tay (da loai mat va cat co tay)
            bbox, mask = detect_hand_bbox(frame, face_cascade, previous_bbox)

            if bbox is not None:
                previous_bbox = bbox
                missed_frames = 0
                x1, y1, x2, y2 = bbox
                # Ve bounding box xanh la bao quanh dung ban tay
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Du doan cu chi moi 0.08 giay de toi uu do muot ma khong giat lag
                if now - last_prediction_time >= 0.08:
                    hand_crop = frame[y1:y2, x1:x2]
                    pred_label, pred_conf, _ = predict_crop(hand_crop, model, scaler)
                    if pred_label is not None:
                        label, confidence, prediction_history = stabilize_prediction(
                            label, confidence, prediction_history, pred_label, pred_conf
                        )
                        last_prediction_time = now

                # Hien thi thong tin cu chi
                if label:
                    cv2.putText(
                        display,
                        f"Gesture: {label.upper()}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
                if confidence is not None:
                    cv2.putText(
                        display,
                        f"Confidence: {confidence:.1%}",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
            else:
                # Khong tim thay ban tay hop le
                missed_frames += 1
                if missed_frames >= 8:
                    previous_bbox = None
                label = None
                confidence = None
                prediction_history = []
                cv2.putText(
                    display,
                    "No hand detected",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            # Hien thi FPS
            cv2.putText(
                display,
                f"FPS: {int(round(fps))}",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # Huong dan phim tat
            cv2.putText(
                display,
                "Press 'q' to quit ",
                (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Gesture webcam", display)

            if show_mask_window:
                cv2.imshow("Skin Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Gesture webcam", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                show_mask_window = not show_mask_window
                if not show_mask_window:
                    cv2.destroyWindow("Skin Mask")

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam demo nhan dien cu chi tay tu dong bang OpenCV + HOG + SVM"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (mac dinh: 0)")
    args = parser.parse_args()
    try:
        run(args.camera)
    except (OSError, ValueError, cv2.error) as exc:
        raise SystemExit(f"Loi: {exc}") from exc