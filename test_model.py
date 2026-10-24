"""Buoc 9: load model va du doan lai tu anh goc trong tap test."""

import argparse
import csv
import hashlib
import json
import pickle

import cv2
import numpy as np

from feature_extraction import extract_features
from preprocessing import ROOT, preprocess_annotated


def load_artifacts():
    models = ROOT / "models"
    metadata = json.loads((models / "model_metadata.json").read_text(encoding="utf-8"))
    for name, expected in metadata["preprocessing"]["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} khong khop pipeline luc train.")
    artifacts = []
    for name in ("gesture_model.pkl", "scaler.pkl", "class_names.pkl"):
        with (models / name).open("rb") as stream:
            artifacts.append(pickle.load(stream))
    model, scaler, classes = artifacts
    if set(model.classes_) != set(classes) or model.n_features_in_ != 8100:
        raise ValueError("Model/class_names khong khop HOG hien tai.")
    return model, scaler, classes


def demo(count=8, seed=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, scaler, _ = load_artifacts()
    results = ROOT / "outputs" / "results"
    with np.load(results / "split_indices.npz", allow_pickle=False) as split:
        test_indices = split["test"]
        if set(test_indices) & (set(split["train"]) | set(split["validation"])):
            raise ValueError("Tap test bi trung voi train/validation.")
    chosen = np.random.default_rng(seed).choice(test_indices, size=min(count, len(test_indices)), replace=False)
    with np.load(results / "hog_features.npz", allow_pickle=False) as data:
        paths, labels, boxes = data["image_paths"][chosen], data["y"][chosen], data["bboxes"][chosen]
        cached = data["X"][chosen]
    fig, axes = plt.subplots(2, 4, figsize=(15, 11), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    rows = []
    for position, (index, path, actual, bbox) in enumerate(zip(chosen, paths, labels, boxes)):
        axis = axes.flat[position]
        try:
            stages = preprocess_annotated(cv2.imread(str(ROOT / str(path))), bbox, mask_method="none")
            vector, _ = extract_features(stages, mode="hog")
            if not np.allclose(vector, cached[position], atol=1e-6, rtol=1e-5):
                raise ValueError("HOG tu anh goc khong khop cache luc train.")
            scaled = scaler.transform(vector.reshape(1, -1))
            predicted = str(model.predict(scaled)[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                # Xac suat cua NHAN predict, khong lay max xac suat nhan khac.
                class_index = list(model.classes_).index(predicted)
                confidence = float(model.predict_proba(scaled)[0, class_index])
            title = f"Thực tế: {actual}\nDự đoán: {predicted}"
            if confidence is not None:
                title += f"\nĐộ tin cậy: {confidence:.1%}"
            axis.imshow(cv2.cvtColor(stages["resized"], cv2.COLOR_BGR2RGB))
            axis.set_title(title, color="green" if predicted == actual else "red", fontsize=11)
            rows.append({"index": int(index), "image": str(path), "actual": str(actual),
                         "predicted": predicted, "confidence": confidence, "error": ""})
            print(f"{position + 1}. Actual={actual}, predicted={predicted}, "
                  f"confidence={confidence:.1%}" if confidence is not None else
                  f"{position + 1}. Actual={actual}, predicted={predicted}")
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            axis.set_title(f"Ảnh lỗi: {position + 1}", color="red")
            rows.append({"index": int(index), "image": str(path), "actual": str(actual),
                         "predicted": "", "confidence": "", "error": str(exc)})
            print(f"ERROR {path}: {exc}")
    success = [row for row in rows if not row["error"]]
    correct = sum(row["actual"] == row["predicted"] for row in success)
    selection = "Ngẫu nhiên mỗi lần chạy" if seed is None else f"seed={seed}"
    fig.suptitle(f"Demo ảnh test | {selection} | Đúng {correct}/{len(success)} ảnh đọc được")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.86,
                        hspace=0.65, wspace=0.15)
    output = ROOT / "outputs" / "figures" / "test_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    with (results / "step9_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "image", "actual", "predicted", "confidence", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Valid: {len(success)}/{len(rows)} | Correct: {correct}/{len(success)}")
    print("HOG recomputed from images matches cached features for valid samples.")
    print("Saved:", output)
    print("Dung o buoc 9; chua mo webcam.")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo du doan 5-8 anh test, khong train lai")
    parser.add_argument("--count", type=int, choices=range(5, 9), default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="Tuy chon: co dinh bo anh de tai lap; mac dinh random moi lan")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    try:
        demo(args.count, args.seed, show=not args.no_show)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
