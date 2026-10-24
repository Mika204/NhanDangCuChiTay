"""Buoc 1: chi doc va kiem tra cau truc dataset, khong sua du lieu."""

import json
from collections import Counter
from pathlib import Path


SELECTED_CLASSES = ["fist", "palm", "like", "dislike", "peace", "ok", "one", "stop"]
ROOT = Path(__file__).resolve().parent


def main():
    annotation_dir = ROOT / "ann_train_val"
    image_dir = ROOT / "hagrid_30k"
    for directory in (annotation_dir, image_dir):
        if not directory.is_dir():
            raise SystemExit(f"Khong tim thay thu muc: {directory}")

    print("Annotation files:", len(list(annotation_dir.glob("*.json"))))
    print("Image folders:", len([p for p in image_dir.iterdir() if p.is_dir()]))
    print("class | images | annotations | matched | missing_annotation | bad_records")
    example = None
    for label in SELECTED_CLASSES:
        annotation_path = annotation_dir / f"{label}.json"
        folder = image_dir / f"train_val_{label}"
        if not annotation_path.is_file() or not folder.is_dir():
            print(f"{label}: thieu JSON hoac thu muc anh")
            continue
        with annotation_path.open(encoding="utf-8") as stream:
            annotations = json.load(stream)
        if not isinstance(annotations, dict):
            print(f"{label}: JSON khong phai dictionary")
            continue
        images = sorted(p for p in folder.iterdir() if p.is_file())
        matched = [p for p in images if p.stem in annotations]
        bad_records = 0
        label_counts = Counter()
        for path in matched:
            record = annotations[path.stem]
            boxes = record.get("bboxes", [])
            labels = record.get("labels", [])
            valid = bool(boxes) and len(boxes) == len(labels)
            for box in boxes:
                valid = valid and isinstance(box, list) and len(box) == 4
                if isinstance(box, list) and len(box) == 4:
                    valid = valid and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in box)
            bad_records += not valid
            label_counts.update(labels)
        print(f"{label} | {len(images)} | {len(annotations)} | {len(matched)} | "
              f"{len(images) - len(matched)} | {bad_records}")
        print("  Extensions:", dict(Counter(p.suffix for p in images)))
        print("  Labels on matched images:", dict(label_counts))
        if example is None and matched:
            path = matched[0]
            example = (path, annotations[path.stem])

    if example:
        path, record = example
        print("\nExample image:", path.relative_to(ROOT))
        print("Annotation key:", path.stem)
        print(json.dumps(record, indent=2, ensure_ascii=False))
    print("\nChi kiem tra cau truc; chua crop, tien xu ly hay train model.")


if __name__ == "__main__":
    main()
