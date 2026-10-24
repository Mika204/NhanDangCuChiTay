"""Chuan bi HOG (buoc 6) va train/so sanh validation (buoc 7)."""

import csv
import argparse
import hashlib
import json
import random
import time
import pickle

import cv2
import numpy as np

from feature_extraction import extract_features
from inspect_dataset import SELECTED_CLASSES
from preprocessing import ROOT, preprocess_annotated


def prepare_features():
    rng = random.Random(42)
    limit = 800
    output = ROOT / "outputs" / "results"
    output.mkdir(parents=True, exist_ok=True)
    # Cap phat truoc de tranh giu nhieu ban sao ma tran trong RAM.
    X = np.empty((len(SELECTED_CLASSES) * limit, 8100), dtype=np.float32)
    samples, errors, summary = [], [], []
    started = time.perf_counter()
    for label in SELECTED_CLASSES:
        with (ROOT / "ann_train_val" / f"{label}.json").open(encoding="utf-8") as stream:
            annotations = json.load(stream)
        candidates = [
            path for path in sorted((ROOT / "hagrid_30k" / f"train_val_{label}").glob("*.jpg"))
            if label in annotations.get(path.stem, {}).get("labels", [])
        ]
        chosen = rng.sample(candidates, min(limit, len(candidates)))
        success = 0
        print(f"{label}: selected {len(chosen)} / {len(candidates)}", flush=True)
        for index, path in enumerate(chosen, 1):
            record = annotations[path.stem]
            try:
                labels, boxes = record["labels"], record["bboxes"]
                if len(labels) != len(boxes):
                    raise ValueError("So labels va boxes khong khop.")
                bbox = boxes[labels.index(label)]
                stages = preprocess_annotated(cv2.imread(str(path)), bbox, mask_method="none")
                vector, _ = extract_features(stages, mode="hog")
                if vector.shape != (8100,) or not np.isfinite(vector).all():
                    raise ValueError("HOG khong hop le.")
                if not np.any(vector):
                    raise ValueError("HOG la vector toan 0.")
                X[len(samples)] = vector
                samples.append({"image": path.relative_to(ROOT).as_posix(), "label": label,
                                "bbox": bbox, "user_id": record.get("user_id", "")})
                success += 1
            except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
                errors.append({"image": path.relative_to(ROOT).as_posix(), "label": label,
                               "error": str(exc)})
            if index % 200 == 0 or index == len(chosen):
                print(f"  processed {index}/{len(chosen)}, valid={success}", flush=True)
        summary.append({"class": label, "selected": len(chosen), "valid": success,
                        "errors": len(chosen) - success})

    if not samples:
        raise ValueError("Khong co mau hop le; khong luu feature cache.")
    X = X[:len(samples)]
    cache = output / "hog_features.npz"
    # Khong nen: ghi/doc nhanh hon, khong can mang Python object/pickle.
    np.savez(cache, X=X, y=np.array([s["label"] for s in samples]),
             image_paths=np.array([s["image"] for s in samples]),
             bboxes=np.array([s["bbox"] for s in samples], dtype=np.float64),
             user_ids=np.array([s["user_id"] for s in samples]),
             class_names=np.array(SELECTED_CLASSES))
    metadata = {"feature_mode": "hog", "random_state": 42, "max_images_per_class": limit,
                "image_size": [128, 128], "feature_count": 8100,
                "pipeline": "bbox crop -> resize -> grayscale -> Gaussian 5x5 -> CLAHE -> HOG",
                "hog": {"cell": [8, 8], "block_cells": [2, 2], "stride_cells": 1,
                        "bins": 9, "normalization": "L2-Hys"},
                "opencv_version": cv2.__version__, "numpy_version": np.__version__,
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("preprocessing.py", "feature_extraction.py")},
                "summary": summary, "samples": len(samples),
                "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (output / "hog_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "hog_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "error"])
        writer.writeheader()
        writer.writerows(errors)
    print("\nclass | selected | valid | errors")
    for row in summary:
        print(f"{row['class']} | {row['selected']} | {row['valid']} | {row['errors']}")
    print("X shape:", X.shape, "dtype:", X.dtype)
    print("Saved:", cache)
    print("Elapsed seconds:", metadata["elapsed_seconds"])
    print("Dung o buoc 6; chua chia tap, fit scaler hay train model.")


def train_models():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except ImportError as exc:
        raise SystemExit("Thieu scikit-learn. Chay: .\\.venv\\Scripts\\python.exe -m pip install scikit-learn") from exc

    output = ROOT / "outputs" / "results"
    cache = output / "hog_features.npz"
    metadata = json.loads((output / "hog_metadata.json").read_text(encoding="utf-8"))
    for name, expected in metadata["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} da thay doi. Chay --prepare de tao lai HOG truoc.")
    with np.load(cache, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        user_ids = data["user_ids"]
        class_names = data["class_names"]
    if X.shape != (len(y), 8100) or not np.isfinite(X).all():
        raise ValueError("Feature cache khong hop le.")
    indices = np.arange(len(y))
    train_idx, rest_idx = train_test_split(indices, test_size=0.30, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50,
                                         stratify=y[rest_idx], random_state=42)
    print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}", flush=True)
    # Theo ke hoach: stratify theo nhan anh, khong chia theo nguoi.
    # Ghi lai so user trung de bao cao dung gioi han cua phep danh gia.
    groups = [set(user_ids[idx]) - {""} for idx in (train_idx, val_idx, test_idx)]
    overlap = {"train_validation": len(groups[0] & groups[1]),
               "train_test": len(groups[0] & groups[2]),
               "validation_test": len(groups[1] & groups[2])}
    print("Shared user_ids between splits:", overlap, flush=True)
    np.savez(output / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    del X
    rows = []
    best_knn = None
    best_knn_accuracy = -1

    def evaluate(name, model):
        started = time.perf_counter()
        print(f"Training {name} ...", flush=True)
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, prediction, average="macro", zero_division=0)
        row = {"model": name, "accuracy": float(accuracy_score(y_val, prediction)),
               "precision_macro": float(precision), "recall_macro": float(recall),
               "f1_macro": float(f1), "seconds": round(time.perf_counter() - started, 2)}
        rows.append(row)
        print(f"  Validation accuracy={row['accuracy']:.4f}, F1={row['f1_macro']:.4f}, "
              f"seconds={row['seconds']}", flush=True)
        return row

    for k in (3, 5, 7):
        model = KNeighborsClassifier(n_neighbors=k, algorithm="brute", n_jobs=1)
        row = evaluate(f"KNN k={k}", model)
        if row["accuracy"] > best_knn_accuracy:
            best_knn_accuracy, best_knn = row["accuracy"], model
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              random_state=42, cache_size=256)
    svm_row = evaluate("SVM rbf", svm)
    best_name = "SVM rbf" if svm_row["accuracy"] > best_knn_accuracy else f"KNN k={best_knn.n_neighbors}"
    # Checkpoint buoc 7: buoc 8 se doc lai, khong can train lai.
    checkpoint = {"knn": best_knn, "svm": svm, "scaler": scaler,
                  "class_names": class_names.tolist(), "best_model": best_name,
                  "validation_results": rows, "shared_user_ids": overlap,
                  "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                  "metadata": metadata}
    with (output / "step7_checkpoint.pkl").open("wb") as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (output / "validation_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Selected by validation accuracy:", best_name)
    print("Checkpoint:", output / "step7_checkpoint.pkl")
    print("Dung o buoc 7; chua du doan tren test hay xuat model cuoi cung.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuan bi HOG hoac train KNN/SVM")
    parser.add_argument("--prepare", action="store_true", help="Buoc 6: tao lai feature cache")
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare_features()
        else:
            train_models()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
