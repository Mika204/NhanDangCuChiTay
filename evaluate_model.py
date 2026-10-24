"""Buoc 8: danh gia test va xuat model da chon bang validation."""

import csv
import hashlib
import json
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from preprocessing import ROOT


def main():
    results = ROOT / "outputs" / "results"
    figures = ROOT / "outputs" / "figures"
    models = ROOT / "models"
    cache = results / "hog_features.npz"
    with (results / "step7_checkpoint.pkl").open("rb") as stream:
        checkpoint = pickle.load(stream)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != checkpoint["cache_sha256"]:
        raise ValueError("Feature cache da thay doi sau khi train. Can train lai.")
    for name, expected in checkpoint["metadata"]["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} da thay doi so voi checkpoint.")
    with np.load(cache, allow_pickle=False) as data:
        y = data["y"]
        with np.load(results / "split_indices.npz", allow_pickle=False) as split:
            train_idx, val_idx, test_idx = split["train"], split["validation"], split["test"]
        # Xac nhan split van dung nhu buoc 7 truoc khi doc dac trung test.
        expected_train, rest = train_test_split(np.arange(len(y)), test_size=0.30,
                                                stratify=y, random_state=42)
        expected_val, expected_test = train_test_split(rest, test_size=0.50,
                                                      stratify=y[rest], random_state=42)
        if not all(np.array_equal(a, b) for a, b in zip(
                (train_idx, val_idx, test_idx), (expected_train, expected_val, expected_test))):
            raise ValueError("Split khong khop voi buoc 7.")
        X_test = checkpoint["scaler"].transform(data["X"][test_idx])
        y_test = y[test_idx]
        paths = data["image_paths"][test_idx]
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    classes = checkpoint["class_names"]
    selected = checkpoint["best_model"]
    candidates = {f"KNN k={checkpoint['knn'].n_neighbors}": checkpoint["knn"],
                  "SVM rbf": checkpoint["svm"]}
    if selected not in candidates:
        raise ValueError("Ten model duoc chon khong hop le.")
    rows, predictions = [], {}
    print("Test samples:", len(y_test))
    print("Model selected by VALIDATION:", selected)
    for name, model in candidates.items():
        prediction = model.predict(X_test)
        predictions[name] = prediction
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="macro", zero_division=0)
        rows.append({"model": name, "accuracy": float(accuracy_score(y_test, prediction)),
                     "precision_macro": float(precision), "recall_macro": float(recall),
                     "f1_macro": float(f1)})
        print(f"{name}: accuracy={rows[-1]['accuracy']:.4f}, F1 macro={f1:.4f}")
    report = classification_report(y_test, predictions[selected], labels=classes,
                                   digits=4, zero_division=0)
    print("\nClassification report -", selected)
    print(report)
    (results / "classification_report.txt").write_text(
        f"Selected by validation: {selected}\n\n{report}", encoding="utf-8")
    with (results / "test_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (results / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "actual", "predicted"])
        writer.writerows(zip(paths, y_test, predictions[selected]))
    matrix = confusion_matrix(y_test, predictions[selected], labels=classes)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test confusion matrix: {selected}")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(metrics))
    for index, row in enumerate(rows):
        bars = ax.bar(positions + (index - 0.5) * 0.35,
                      [row[key] for key in metrics], width=0.35, label=row["model"])
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_xticks(positions, ["Accuracy", "Precision macro", "Recall macro", "F1 macro"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("KNN vs SVM on test (model selection used validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "knn_vs_svm.png", dpi=150)
    plt.close(fig)
    for filename, value in [("gesture_model.pkl", candidates[selected]),
                            ("scaler.pkl", checkpoint["scaler"]),
                            ("class_names.pkl", classes)]:
        with (models / filename).open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {"selected_by_validation": selected, "test_metrics": rows,
                "preprocessing": checkpoint["metadata"],
                "shared_user_ids": checkpoint["shared_user_ids"],
                "evaluation_limit": "Image-stratified split; users may overlap between splits."}
    (models / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Kiem tra artifact vua xuat dung duoc ma khong fit lai.
    with (models / "gesture_model.pkl").open("rb") as stream:
        loaded_model = pickle.load(stream)
    if not np.array_equal(loaded_model.predict(X_test[:8]), predictions[selected][:8]):
        raise ValueError("Model doc lai du doan khong khop.")
    print("Saved models:", models)
    print("Saved figures:", figures)
    print("Shared user_ids:", checkpoint["shared_user_ids"])
    print("Ket qua chua dai dien hoan toan cho nguoi moi.")
    print("Dung o buoc 8; chua viet demo test anh hay webcam.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Loi: {exc}") from exc
