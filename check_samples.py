"""Buoc 5: kiem tra 10 anh ngau nhien moi class, chua train."""

import argparse
import csv
import json
import random

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_extraction import extract_features
from inspect_dataset import SELECTED_CLASSES
from preprocessing import ROOT, preprocess_annotated


def main(mask_method="ycrcb", feature_mode="hog"):
    rng = random.Random(42)
    run_name = "hog" if feature_mode == "hog" else mask_method
    figures = ROOT / "outputs" / "figures" / f"step5_{run_name}"
    results = ROOT / "outputs" / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Features:", feature_mode)
    print("Mask method:", "none" if feature_mode == "hog" else mask_method)
    print("class | sampled | feature_ok | errors | review_flags")
    for label in SELECTED_CLASSES:
        with (ROOT / "ann_train_val" / f"{label}.json").open(encoding="utf-8") as stream:
            annotations = json.load(stream)
        folder = ROOT / "hagrid_30k" / f"train_val_{label}"
        candidates = [
            p for p in sorted(folder.glob("*.jpg"))
            if label in annotations.get(p.stem, {}).get("labels", [])
        ]
        chosen = rng.sample(candidates, min(10, len(candidates)))
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        for axis in axes.flat:
            axis.axis("off")
        ok = flags = 0
        for index, path in enumerate(chosen):
            row = {"class": label, "image": str(path.relative_to(ROOT)),
                   "status": "error", "features": "", "mask_ratio": "",
                   "largest_contour_ratio": "", "review": "", "error": ""}
            # Hai hang cho moi nhom 5 anh: crop, roi mask + contour.
            top = axes[(index // 5) * 2, index % 5]
            bottom = axes[(index // 5) * 2 + 1, index % 5]
            short_id = f"{index + 1}: {path.stem[:8]}"
            try:
                record = annotations[path.stem]
                labels, boxes = record["labels"], record["bboxes"]
                if len(labels) != len(boxes):
                    raise ValueError("So labels va boxes khong khop.")
                stages = preprocess_annotated(cv2.imread(str(path)), boxes[labels.index(label)],
                                              mask_method="none" if feature_mode == "hog" else mask_method)
                top.imshow(cv2.cvtColor(stages["resized"], cv2.COLOR_BGR2RGB))
                top.set_title(short_id)
                vector, parts = extract_features(stages, mode=feature_mode)
                expected_size = 8100 if feature_mode == "hog" else 8112
                if vector.shape != (expected_size,) or not np.isfinite(vector).all():
                    raise ValueError("Feature vector sai kich thuoc hoac khong huu han.")
                if feature_mode == "hog":
                    bottom.imshow(stages["clahe"], cmap="gray", vmin=0, vmax=255)
                    review = not np.any(vector)
                    bottom.set_title("HOG: 8100" + (" | ZERO VECTOR" if review else ""))
                    row.update(status="ok", features=vector.size, review=review)
                    ok += 1
                    flags += int(review)
                    rows.append(row)
                    continue
                bottom.imshow(stages["cleaned"], cmap="gray", vmin=0, vmax=255)
                mask_ratio = np.count_nonzero(stages["cleaned"]) / stages["cleaned"].size
                contour_ratio = float(parts["geometry"][0])
                # Chi la co hieu de xem lai, khong tu dong loai anh.
                review = mask_ratio < 0.10 or mask_ratio > 0.90 or contour_ratio < 0.10
                contour = parts["contour"][:, 0, :]
                bottom.plot(contour[:, 0], contour[:, 1], color="lime", linewidth=0.8)
                bottom.set_title(f"mask={mask_ratio:.2f}" + (" | REVIEW" if review else ""))
                row.update(status="ok", features=vector.size, mask_ratio=mask_ratio,
                           largest_contour_ratio=contour_ratio, review=review)
                ok += 1
                flags += int(review)
            except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
                row["error"] = str(exc)
                bottom.set_title("ERROR", color="red")
                print(f"  {label}/{path.name}: {exc}")
            rows.append(row)
        bottom_name = "CLAHE below" if feature_mode == "hog" else "mask below"
        fig.suptitle(f"{label}: 10 samples | crop above, {bottom_name} | {run_name} | seed=42")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(figures / f"{label}.png", dpi=120)
        plt.close(fig)
        print(f"{label} | {len(chosen)} | {ok} | {len(chosen) - ok} | {flags}")

    report = results / f"step5_{run_name}_samples.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "image", "status", "features",
            "mask_ratio", "largest_contour_ratio", "review", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print("Report:", report)
    print("Figures:", figures)
    print("HOG OK chi xac nhan vector hop le, chua danh gia kha nang phan loai."
          if feature_mode == "hog" else "Feature OK khong dam bao mask dung.")
    print("Dung o buoc 5; chua xu ly 800 anh/class hay train model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiem tra 10 anh/class, seed=42")
    parser.add_argument("--mask", choices=["ycrcb", "otsu", "grabcut"], default="ycrcb")
    parser.add_argument("--features", choices=["hog", "combined"], default="hog")
    args = parser.parse_args()
    try:
        main(mask_method=args.mask, feature_mode=args.features)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
