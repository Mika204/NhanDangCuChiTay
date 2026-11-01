"""Webcam demo tu dong tim va crop CHINH XAC vung ban tay bang xu ly anh OpenCV truyen thong
(Ket hop YCrCb + HSV, loc mat, loai bo cang tay/canh tay) va du doan cu chi bang HOG + SVM.
"""

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

    # 2. HSV: Loc them theo sac do (Hue 0-25) va do bao hoa (Sat 30-200) de loai bo tuong vang va nen go
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, (0, 30, 60), (25, 200, 255))

    # Giao hai mask de co vung mau da that su tin cay
    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # 3. Phep toan hinh thai hoc: Opening khu nhieu hat, Closing lam lien khoi ban tay
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def isolate_hand_from_arm(cnt, frame_w, frame_h):
    """Chi crop dung phan ban tay (tu dau ngon tay den co tay), cat bo phan cang tay keo dai."""
    x, y, w, h = cv2.boundingRect(cnt)

    # Ty le chieu dai / chieu rong cua ban tay nguoi tu co tay den dau ngon tay thuong la 1.0 - 1.25.
    # Neu chieu cao h vuot qua 1.25 * w, phan du ra ben duoi la cang tay/canh tay.
    arm_from_bottom = (y + h) >= (frame_h - 40) or (h > 1.25 * w)
    arm_from_right = (x + w) >= (frame_w - 30) and (w > 1.25 * h)
    arm_from_left = (x <= 30) and (w > 1.25 * h)

    if arm_from_bottom:
        # Canh tay dua tu duoi len: ban tay o tren cung, co tay va cang tay o duoi
        hand_h = min(h, int(w * 1.25))
        # Gioi han lai chieu cao chi lay dung ban tay tu dinh ngon tay xuong
        h = max(20, hand_h)
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
    pad_x = int(w * 0.07)
    pad_y = int(h * 0.07)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
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


def detect_hand_bbox(frame):
    """Tim contour ban tay phu hop nhat va tra ve bounding box chi bao quanh ban tay."""
    height, width = frame.shape[:2]
    mask = segment_skin(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best_score = 0.0
    best_cnt = None

    for cnt in contours:
        score = score_hand_candidate(cnt, width, height)
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None or best_score < 2000.0:
        return None, mask

    # Tinh bounding box va cat bo phan cang tay
    bbox = isolate_hand_from_arm(best_cnt, width, height)
    x1, y1, x2, y2 = bbox

    if (x2 - x1) < 25 or (y2 - y1) < 25:
        return None, mask

    return (x1, y1, x2, y2), mask


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


def run(camera=0):
    model, scaler, _ = load_artifacts()
    capture = cv2.VideoCapture(camera)

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
        print("- Phim 'm': Bat/tat cua so hien thi Mask phan doan da.")

        show_mask_window = False
        label = None
        confidence = None
        last_prediction_time = 0.0

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
            bbox, mask = detect_hand_bbox(frame)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Ve bounding box xanh la bao quanh dung ban tay
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Du doan cu chi moi 0.08 giay de toi uu do muot ma khong giat lag
                if now - last_prediction_time >= 0.08:
                    hand_crop = frame[y1:y2, x1:x2]
                    pred_label, pred_conf, _ = predict_crop(hand_crop, model, scaler)
                    if pred_label is not None:
                        label = pred_label
                        confidence = pred_conf
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
                label = None
                confidence = None
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
                "Press 'q' to quit | 'm' to toggle mask",
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
