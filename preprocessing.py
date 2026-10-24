"""Thu crop va hien thi pipeline tien xu ly tren mot anh."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def crop_hand(image, bbox):
    """HaGRID bbox: [x, y, width, height], chuan hoa theo kich thuoc anh."""
    if image is None or image.size == 0:
        raise ValueError("Anh rong hoac khong doc duoc.")
    if len(bbox) != 4 or not all(
        isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
        for v in bbox
    ):
        raise ValueError(f"Bounding box khong hop le: {bbox}")
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError("Chieu rong va chieu cao box phai lon hon 0.")
    image_height, image_width = image.shape[:2]
    x1 = max(0, math.floor(x * image_width))
    y1 = max(0, math.floor(y * image_height))
    x2 = min(image_width, math.ceil((x + width) * image_width))
    y2 = min(image_height, math.ceil((y + height) * image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box khong giao voi anh.")
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def preprocess_annotated(image, bbox, mask_method="ycrcb"):
    """GrabCut dung nen ngoai bbox; dau ra van la crop bbox 128x128."""
    import cv2
    import numpy as np

    crop, (x1, y1, x2, y2) = crop_hand(image, bbox)
    if mask_method != "grabcut":
        return preprocess_hand(crop, mask_method)
    height, width = image.shape[:2]
    margin = max(8, math.ceil(max(x2 - x1, y2 - y1) * 0.2))
    left, top = max(0, x1 - margin), max(0, y1 - margin)
    right, bottom = min(width, x2 + margin), min(height, y2 + margin)
    context = image[top:bottom, left:right]
    scale = min(1.0, 256 / max(context.shape[:2]))
    context = cv2.resize(context, (max(1, round(context.shape[1] * scale)),
                                   max(1, round(context.shape[0] * scale))))
    sx = context.shape[1] / (right - left)
    sy = context.shape[0] / (bottom - top)
    bx1, by1 = math.floor((x1 - left) * sx), math.floor((y1 - top) * sy)
    bx2, by2 = math.ceil((x2 - left) * sx), math.ceil((y2 - top) * sy)
    labels = np.zeros(context.shape[:2], dtype=np.uint8)
    if (bx2 - bx1) * (by2 - by1) >= labels.size:
        raise ValueError("Khong co vung nen ngoai bbox de khoi tao GrabCut.")
    labels[by1:by2, bx1:bx2] = cv2.GC_PR_FGD
    # Gia thuyet thu nghiem: vung nho giua bbox nam tren long ban tay.
    # Giu foreground mau de tranh GrabCut loai sach doi tuong.
    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
    radius = max(1, min(bx2 - bx1, by2 - by1) // 20)
    labels[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] = cv2.GC_FGD
    cv2.setRNGSeed(42)
    cv2.grabCut(context, labels, (bx1, by1, bx2 - bx1, by2 - by1),
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)
    mask = cv2.resize(mask[by1:by2, bx1:bx2], (128, 128), interpolation=cv2.INTER_NEAREST)
    stages = preprocess_hand(crop, "ycrcb")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    stages.update(mask=mask, cleaned=cleaned, mask_method="grabcut", threshold=None)
    return stages


def preprocess_hand(crop, mask_method="ycrcb"):
    """Tra ve cac anh trung gian; mask trang la vung tay du kien."""
    import cv2

    if crop is None or crop.size == 0:
        raise ValueError("Anh crop rong.")
    resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)
    if mask_method == "none":
        return {"resized": resized, "blurred": blurred, "clahe": clahe}
    if mask_method == "otsu":
        threshold, mask = cv2.threshold(
            clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    elif mask_method == "ycrcb":
        # Nhanh mask dung mau, khong threshold anh xam sau CLAHE.
        # Nguong thu nghiem, can kiem tra lai tren nhieu anh.
        color_blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        ycrcb = cv2.cvtColor(color_blurred, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        threshold = None
    elif mask_method == "grabcut":
        raise ValueError("GrabCut can anh goc va bbox: dung preprocess_annotated.")
    else:
        raise ValueError(f"Phuong phap mask khong hop le: {mask_method}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return {
        "resized": resized,
        "blurred": blurred,
        "clahe": clahe,
        "mask": mask,
        "cleaned": cleaned,
        "threshold": threshold,
        "mask_method": mask_method,
    }


def show_preprocessing(original, stages, output, show=True):
    import cv2
    import matplotlib.pyplot as plt

    panels = [
        (original, "Original + bounding box"),
        (stages["resized"], "Crop + resize 128 x 128"),
        (stages["blurred"], "Grayscale + Gaussian Blur"),
        (stages["clahe"], "CLAHE"),
        (stages["mask"], f"Mask: {stages['mask_method']}"),
        (stages["cleaned"], "Opening + Closing"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (image, title) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print("Saved:", output)
    if show:
        plt.show()
    plt.close(fig)


def demo_crop(step="pipeline", show=True, mask_method="ycrcb"):
    try:
        import cv2
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Thieu thu vien. Chay: .\\.venv\\Scripts\\python.exe -m pip install "
            "opencv-python matplotlib"
        ) from exc

    label = "fist"
    annotation_path = ROOT / "ann_train_val" / f"{label}.json"
    folder = ROOT / "hagrid_30k" / f"train_val_{label}"
    with annotation_path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)

    for path in sorted(folder.glob("*.jpg")):
        record = annotations.get(path.stem)
        if not record or label not in record.get("labels", []):
            continue
        labels = record["labels"]
        boxes = record.get("bboxes", [])
        if len(labels) != len(boxes):
            print("Bo qua annotation sai:", path.name)
            continue
        # Lay box theo nhan, khong mac dinh lay box dau tien.
        bbox = boxes[labels.index(label)]
        image = cv2.imread(str(path))
        try:
            crop, (x1, y1, x2, y2) = crop_hand(image, bbox)
        except (ValueError, TypeError) as exc:
            print(f"Bo qua {path.name}: {exc}")
            continue

        print("Image:", path.relative_to(ROOT))
        print("Label:", label)
        print("Normalized bbox [x, y, width, height]:", bbox)
        print("Image size (width, height):", (image.shape[1], image.shape[0]))
        print("Pixel bbox [x1, y1, x2, y2]:", (x1, y1, x2, y2))
        print("Crop size (width, height):", (crop.shape[1], crop.shape[0]))

        marked = image.copy()
        cv2.rectangle(marked, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 3)
        if step == "pipeline":
            stages = preprocess_annotated(image, bbox, mask_method=mask_method)
            print("Mask method:", stages["mask_method"])
            if stages["threshold"] is not None:
                print("Otsu threshold:", stages["threshold"])
            print("Processed size:", stages["cleaned"].shape)
            output = ROOT / "outputs" / "figures" / f"step3_{mask_method}.png"
            show_preprocessing(marked, stages, output, show=show)
            return
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original + bounding box")
        axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Cropped hand: {label}")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        output = ROOT / "outputs" / "figures" / "step2_crop.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print("Saved:", output)
        if show:
            plt.show()
        plt.close(fig)
        return
    raise ValueError("Khong tim thay anh fist doc duoc voi annotation hop le.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo crop va tien xu ly mot anh fist")
    parser.add_argument("--step", choices=["crop", "pipeline"], default="pipeline")
    parser.add_argument("--no-show", action="store_true", help="Chi luu hinh")
    parser.add_argument("--mask", choices=["ycrcb", "otsu", "grabcut"], default="ycrcb")
    args = parser.parse_args()
    try:
        demo_crop(step=args.step, show=not args.no_show, mask_method=args.mask)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
