"""HOG la mac dinh; Hu Moments va hinh hoc danh cho thu nghiem combined."""

import argparse
import json

import cv2
import numpy as np

from preprocessing import ROOT, crop_hand, preprocess_hand


def extract_hog(gray):
    """HOG 9 bins, cell 8x8, block 2x2 cells, stride 1 cell, L2-Hys."""
    if gray is None or gray.shape != (128, 128) or gray.dtype != np.uint8:
        raise ValueError("HOG can anh xam uint8 kich thuoc 128 x 128.")
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    position = (angle % 180) / 20
    lower = np.floor(position).astype(int)
    fraction = position - lower
    rows, cols = np.indices(gray.shape)
    hist = np.zeros((16, 16, 9), dtype=np.float32)
    np.add.at(hist, (rows // 8, cols // 8, lower % 9), magnitude * (1 - fraction))
    np.add.at(hist, (rows // 8, cols // 8, (lower + 1) % 9), magnitude * fraction)
    blocks = []
    for row in range(15):
        for col in range(15):
            block = hist[row:row + 2, col:col + 2].ravel().copy()
            block /= np.sqrt(np.sum(block**2) + 1e-10)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.sum(block**2) + 1e-10)
            blocks.append(block)
    return np.concatenate(blocks)


def extract_features(stages, mode="hog"):
    """Mac dinh HOG tren CLAHE; combined them Hu va hinh hoc tu mask."""
    gray = stages["clahe"]
    if mode == "hog":
        hog = extract_hog(gray).astype(np.float32)
        if not np.isfinite(hog).all():
            raise ValueError("HOG chua NaN hoac infinity.")
        return hog, {"hog": hog}
    if mode != "combined":
        raise ValueError(f"Feature mode khong hop le: {mode}")
    mask = stages["cleaned"]
    if gray.shape != (128, 128) or mask.shape != (128, 128):
        raise ValueError("Anh dau vao phai co kich thuoc 128 x 128.")
    if gray.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError("Anh dau vao phai co dtype uint8.")

    # Window 128x128; block 16x16; stride 8x8; cell 8x8; 9 bins.
    hog = extract_hog(gray)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask rong, khong co contour de tinh Hu Moments.")
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if area <= 0 or perimeter <= 0 or hull_area <= 0:
        raise ValueError("Contour suy bien, khong the tinh dac trung hinh hoc.")

    hu_raw = cv2.HuMoments(cv2.moments(contour)).ravel()
    # Nen dai gia tri bang log, giu so 0 la 0.
    hu = -np.sign(hu_raw) * np.log10(np.maximum(np.abs(hu_raw), 1e-30))
    _, _, width, height = cv2.boundingRect(contour)
    geometry = np.array([
        area / mask.size,
        perimeter / (2 * sum(mask.shape)),
        width / height,
        area / hull_area,
        4 * np.pi * area / perimeter**2,
    ], dtype=np.float32)
    vector = np.concatenate([hog, hu, geometry]).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector chua NaN hoac infinity.")
    return vector, {"hog": hog, "hu": hu, "geometry": geometry, "contour": contour}


def demo(show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = "fist"
    with (ROOT / "ann_train_val" / f"{label}.json").open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    for path in sorted((ROOT / "hagrid_30k" / f"train_val_{label}").glob("*.jpg")):
        record = annotations.get(path.stem, {})
        labels = record.get("labels", [])
        boxes = record.get("bboxes", [])
        if label not in labels or len(labels) != len(boxes):
            continue
        try:
            crop, _ = crop_hand(cv2.imread(str(path)), boxes[labels.index(label)])
            stages = preprocess_hand(crop, mask_method="none")
            vector, parts = extract_features(stages, mode="hog")
        except (ValueError, TypeError) as exc:
            print(f"Bo qua {path.name}: {exc}")
            continue

        print("Image:", path.name)
        print("HOG features:", parts["hog"].size)
        print("Total features:", vector.size)
        print("All finite:", bool(np.isfinite(vector).all()))

        overlay = stages["resized"].copy()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Cropped hand (no mask)")
        axes[0].axis("off")
        axes[1].imshow(stages["clahe"], cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("HOG input: CLAHE grayscale")
        axes[1].axis("off")
        axes[2].plot(parts["hog"], linewidth=0.5)
        axes[2].set_title(f"HOG vector ({parts['hog'].size} values)")
        axes[2].set_xlabel("Feature index")
        axes[2].set_ylabel("Value")
        fig.tight_layout()
        output = ROOT / "outputs" / "figures" / "step4_features.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print("Saved:", output)
        if show:
            plt.show()
        plt.close(fig)
        return
    raise ValueError("Khong tim thay anh fist co dac trung hop le.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiem tra feature tren mot anh fist")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    try:
        demo(show=not args.no_show)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
