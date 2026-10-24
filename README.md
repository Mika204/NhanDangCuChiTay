# Nhận diện cử chỉ tay bằng HOG và Machine Learning

> Repository chỉ chứa source code và hướng dẫn. Dataset (`ann_train_val/`, `hagrid_30k/`), model (`models/`) và kết quả sinh ra (`outputs/`) không được đưa lên Git. Khi tải repo về máy mới, cần cung cấp dataset đúng cấu trúc rồi chạy bước 6 → 7 → 8 để tạo model trước khi demo.

## Lệnh chạy chương trình

Mở terminal PowerShell tại `D:\PythonProject`. Các lệnh này dùng model đã lưu, không train lại.

**1. Chạy webcam:**

```powershell
.\.venv\Scripts\python.exe .\main.py webcam
```

Đưa bàn tay trước camera → nhấn **R** → kéo chuột khoanh sát bàn tay → **Enter** để bắt đầu dự đoán. **Space** tạm dừng/chạy tiếp; **Q** thoát.

**2. Xuất ảnh dự đoán trên tập test:**

```powershell
.\.venv\Scripts\python.exe .\test_model.py
```

Lệnh dự đoán 8 ảnh test, mở cửa sổ hiển thị nhãn thật, nhãn dự đoán cùng độ tin cậy, đồng thời lưu hình vào **`outputs/figures/test_predictions.png`**. Chạy lại sẽ ghi đè hình này. Thêm `--no-show` nếu chỉ muốn lưu hình mà không mở cửa sổ.

Đề tài môn **Xử lý ảnh và Thị giác máy tính**, sử dụng OpenCV để xử lý ảnh, HOG để trích xuất đặc trưng và so sánh KNN với SVM để phân loại. Mô hình hiện được chọn là **SVM RBF**. Không sử dụng CNN, TensorFlow hay mô hình Deep Learning.

## 1. Phạm vi và pipeline

Chỉ sử dụng 8 cử chỉ của HaGRID Sample 30k:

```python
["fist", "palm", "like", "dislike", "peace", "ok", "one", "stop"]
```

Lấy ngẫu nhiên tối đa **800 ảnh/class**, với seed **42**. Lần chạy hiện tại có 6.400 ảnh hợp lệ.

```text
Ảnh gốc + bounding box
    → Crop bàn tay
    → Resize 128 × 128
    → Grayscale
    → Gaussian Blur 5 × 5
    → CLAHE
    → HOG: 8.100 giá trị
    → StandardScaler
    → KNN / SVM
    → Nhãn dự đoán
```

HOG dùng cell 8 × 8, block 2 × 2 cell, bước trượt 1 cell, 9 bin hướng và chuẩn hóa L2-Hys. HOG được tính bằng NumPy và các phép gradient của OpenCV.

Hu Moments, đặc trưng hình học và các phương pháp mask vẫn được giữ để thử nghiệm. **Model hiện tại chỉ sử dụng HOG**, không phụ thuộc mask, vì phân đoạn tay trên nền phức tạp chưa ổn định.

## 2. Cấu trúc project

```text
PythonProject/
├── ann_train_val/           # Annotation JSON, giữ nguyên dataset
├── hagrid_30k/              # Ảnh, giữ nguyên dataset
├── models/
│   ├── gesture_model.pkl    # Model được chọn theo validation
│   ├── scaler.pkl
│   ├── class_names.pkl
│   └── model_metadata.json
├── outputs/
│   ├── figures/            # Pipeline, mẫu thử, confusion matrix, biểu đồ
│   └── results/            # HOG cache, split, checkpoint và báo cáo
├── inspect_dataset.py      # Kiểm tra cấu trúc và ánh xạ annotation
├── preprocessing.py        # Crop, tiền xử lý, thử nghiệm mask
├── feature_extraction.py   # HOG mặc định; combined để thử nghiệm
├── check_samples.py        # Kiểm tra 10 ảnh/class
├── train_model.py          # Chuẩn bị HOG hoặc train KNN/SVM
├── evaluate_model.py       # Đánh giá test và xuất model
├── test_model.py           # Dự đoán lại từ ảnh test gốc
├── webcam_demo.py          # Webcam với ROI thủ công
├── main.py                 # Entry point cho demo ảnh/webcam
└── requirements.txt
```

## 3. Môi trường và cài thư viện

Các lệnh dưới đây chạy trong **PowerShell**, tại thư mục project:

```powershell
cd D:\PythonProject
```

Môi trường đã chạy project là **Python 3.14 trên Windows 64-bit**, với các phiên bản thư viện trong `requirements.txt`. Nếu đã có `.venv`, dùng lại môi trường đó; không cần tạo lại.

Chỉ trên máy mới chưa có môi trường, tạo bằng:

```powershell
py -3.14 -m venv .venv
```

Cài dependencies nếu chưa có:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh gọi thẳng Python trong `.venv`, không cần activate môi trường. Trong IDE, chọn interpreter `D:\PythonProject\.venv\Scripts\python.exe`.

## 4. Chạy nhanh với model đã lưu

Nếu đã hoàn thành train và đánh giá, **không cần train lại** khi chạy demo.

### Test ảnh

```powershell
.\.venv\Scripts\python.exe .\main.py test
```

Mỗi lần chạy, lệnh tự chọn ngẫu nhiên 8 ảnh từ tập test, không cố định seed. Ảnh không lặp trong cùng một lần chọn, nhưng có thể xuất hiện lại ở lần chạy khác. Chương trình đọc ảnh gốc, crop theo box đã lưu, chạy lại pipeline HOG, load scaler/model và hiển thị:

- Thực tế, dự đoán và xác suất của nhãn dự đoán.
- Tiêu đề xanh khi đúng, đỏ khi sai.

Hình lưu tại `outputs/figures/test_predictions.png`; kết quả từng ảnh lưu tại `outputs/results/step9_predictions.csv`. Script cũng kiểm tra HOG tính lại có khớp cache lúc train hay không.

Chỉ lưu hình, không mở cửa sổ:

```powershell
.\.venv\Scripts\python.exe .\main.py test --no-show
```

Nếu cần cố định bộ ảnh để tái lập, có thể truyền seed (không bắt buộc):

```powershell
.\.venv\Scripts\python.exe .\test_model.py --count 8 --seed 7
```

`--count` nhận giá trị từ 5 đến 8. Demo ảnh cần model, feature cache `hog_features.npz`, `split_indices.npz` và dataset gốc.

### Webcam

```powershell
.\.venv\Scripts\python.exe .\main.py webcam
```

1. Đưa bàn tay lên trước camera, giữ đủ sáng và hạn chế nền phức tạp.
2. Nhấn **R** trong cửa sổ webcam để chọn ROI.
3. Kéo chuột khoanh sát toàn bộ bàn tay đang hiển thị; tránh lấy mặt hoặc quá nhiều nền.
4. Nhấn **Enter** hoặc **Space** để xác nhận vùng chọn. Chương trình tự bắt đầu dự đoán.
5. Xem `Gesture` và `Confidence` trên cửa sổ webcam.

| Phím | Chức năng |
|---|---|
| R | Chọn lại ROI trên frame đang hiển thị |
| Enter / Space trong cửa sổ chọn ROI | Xác nhận vùng chọn |
| C trong cửa sổ chọn ROI | Hủy chọn, giữ ROI cũ và tạm dừng |
| Space trong cửa sổ webcam | Tạm dừng / chạy tiếp dự đoán |
| Q / Esc trong cửa sổ webcam | Thoát |

ROI đứng yên, **không tự theo dõi bàn tay**. Nếu đổi vị trí hoặc tư thế làm tay vượt khung, nhấn R để khoanh lại. Hình đầu vào không được lật ngang.

Nếu camera mặc định không hoạt động:

```powershell
.\.venv\Scripts\python.exe .\main.py webcam --camera 1
```

Webcam chỉ cần các artifact trong `models/` và source code, không cần feature cache hay dataset. Model chỉ phân loại 8 nhãn; **chưa có bộ phát hiện tay hoặc nhãn “không có tay”**. Nó vẫn có thể đưa ra nhãn khi ROI chứa mặt, nền hoặc vật khác. Confidence là xác suất do model ước lượng, không bảo đảm dự đoán đúng.

## 5. Quy trình từ dataset đến model

Thực hiện từng bước và kiểm tra output trước khi chạy bước sau. Các lệnh chuẩn bị/train/đánh giá sẽ ghi lại các artifact tương ứng; demo chỉ load model, không fit lại.

### Bước 1: Kiểm tra dataset

```powershell
.\.venv\Scripts\python.exe .\inspect_dataset.py
```

Cấu trúc dataset hiện tại:

```text
ann_train_val/fist.json
hagrid_30k/train_val_fist/<ID>.jpg
```

Các class khác có cấu trúc tương tự. JSON là dictionary có khóa ID trùng tên ảnh bỏ phần `.jpg`. `bboxes[i]` tương ứng với `labels[i]`; box có dạng chuẩn hóa `[x, y, width, height]`, với `(x, y)` là góc trên bên trái.

Một ảnh có thể có cả nhãn cử chỉ và `no_gesture`, nên phải chọn box theo nhãn. JSON chứa nhiều bản ghi hơn số ảnh trong bản sample; code chỉ dùng ảnh thực sự có trên đĩa. Không di chuyển hoặc xóa dataset.

### Bước 2–3: Thử crop và tiền xử lý

```powershell
.\.venv\Scripts\python.exe .\preprocessing.py --step crop
.\.venv\Scripts\python.exe .\preprocessing.py --step pipeline
```

Demo pipeline hiển thị cả mask thử nghiệm, mặc định YCrCb. Có thể thêm `--mask otsu` hoặc `--mask grabcut` để so sánh, hoặc `--no-show` để chỉ lưu hình. Các mask này **không tham gia pipeline HOG của model hiện tại**.

### Bước 4: Kiểm tra HOG trên một ảnh

```powershell
.\.venv\Scripts\python.exe .\feature_extraction.py
```

Kết quả mong đợi: 8.100 đặc trưng, `All finite: True`. Hình lưu tại `outputs/figures/step4_features.png`.

### Bước 5: Kiểm tra 10 ảnh/class

```powershell
.\.venv\Scripts\python.exe .\check_samples.py
```

Mặc định dùng HOG, không mask. Lần chạy đã kiểm tra: 80/80 ảnh hợp lệ, không vector toàn 0. Hình lưu tại `outputs/figures/step5_hog/`; báo cáo tại `outputs/results/step5_hog_samples.csv`.

Muốn chạy lại thí nghiệm HOG + Hu + hình học:

```powershell
.\.venv\Scripts\python.exe .\check_samples.py --features combined --mask grabcut
```

Đây chỉ là thí nghiệm phân đoạn; vector hữu hạn và ít cờ `REVIEW` không xác nhận mask đúng.

### Bước 6: Chuẩn bị tối đa 800 ảnh/class

```powershell
.\.venv\Scripts\python.exe .\train_model.py --prepare
```

**Phải có `--prepare` để tạo HOG; chạy không có cờ này sẽ train mô hình.**

Lệnh lấy mẫu seed 42, crop theo annotation, tạo HOG và lưu:

| File trong `outputs/results/` | Nội dung |
|---|---|
| `hog_features.npz` | X, nhãn y, đường dẫn ảnh, box, user_id và danh sách class |
| `hog_metadata.json` | Pipeline, tham số HOG, phiên bản thư viện và hash source |
| `hog_errors.csv` | Các ảnh xử lý lỗi |

Kết quả hiện tại: `X.shape = (6400, 8100)`, `float32`, file cache khoảng 211 MB. Ảnh lỗi được ghi lại và bỏ qua; số mẫu hợp lệ có thể giảm nếu dataset khác. Cần đủ RAM cho cache và các ma trận phục vụ train.

### Bước 7: Train và chọn model bằng validation

```powershell
.\.venv\Scripts\python.exe .\train_model.py
```

Lệnh đọc cache, không trích xuất HOG lại:

- Chia stratified theo nhãn ảnh, seed 42: train 70%, validation 15%, test 15%.
- Với 6.400 ảnh: 4.480 train, 960 validation, 960 test.
- Fit StandardScaler chỉ trên train; transform validation bằng scaler đó.
- Thử KNN K=3, 5, 7 và SVM RBF (`C=1`, `gamma="scale"`, `probability=True`).
- Chọn KNN tốt nhất rồi so sánh với SVM bằng **accuracy validation**; nếu hòa ưu tiên KNN.
- Chưa dự đoán test ở bước này.

Lưu `split_indices.npz`, `validation_results.csv`, `step7_checkpoint.pkl` trong `outputs/results/`. Checkpoint giữ model đã fit và scaler để đánh giá sau mà không train lại.

SVM có tính xác suất nên có thể mất vài phút; lần chạy hiện tại khoảng 408 giây. Cảnh báo `FutureWarning` về `probability` trên scikit-learn 1.9.1 không làm lần train này thất bại. Project đang giữ cách gọi này theo cấu hình đã chạy thành công.

### Bước 8: Đánh giá test và lưu model

```powershell
.\.venv\Scripts\python.exe .\evaluate_model.py
```

Lệnh load checkpoint, kiểm tra cache/pipeline/split, so sánh KNN và SVM trên test, in Classification Report và xuất model đã chọn ở bước 7. **Không chọn lại model theo điểm test và không train lại.**

Các đầu ra:

| Đường dẫn | Nội dung |
|---|---|
| `outputs/results/test_metrics.csv` | Accuracy, Precision/Recall/F1 macro của hai model |
| `outputs/results/classification_report.txt` | Báo cáo từng class của model được chọn |
| `outputs/results/test_predictions.csv` | Dự đoán trên toàn bộ test của model được chọn |
| `outputs/figures/confusion_matrix.png` | Confusion matrix của model được chọn |
| `outputs/figures/knn_vs_svm.png` | Biểu đồ so sánh trên test |
| `models/gesture_model.pkl` | Model dùng cho dự đoán |
| `models/scaler.pkl` | Scaler đã fit trên train |
| `models/class_names.pkl` | Danh sách 8 class |
| `models/model_metadata.json` | Pipeline, kết quả và giới hạn đánh giá |

Sau bước này, chạy demo ảnh/webcam theo mục 4.

## 6. Kết quả đã chạy

Validation:

| Model | Accuracy | F1 macro |
|---|---:|---:|
| KNN K=3 | 93,13% | 93,12% |
| KNN K=5 | 92,71% | 92,74% |
| KNN K=7 | 92,71% | 92,80% |
| **SVM RBF** | **94,48%** | **94,47%** |

Test (960 ảnh, 120 ảnh/class):

| Model | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| KNN K=3 | 91,77% | 92,46% | 91,77% | 91,80% |
| **SVM RBF** | **94,58%** | **94,68%** | **94,58%** | **94,60%** |

Demo seed 42 dự đoán đúng 8/8 ảnh đã chọn; đây là minh họa, không thay thế kết quả trên toàn bộ 960 ảnh test.

## 7. Giới hạn đánh giá

Split hiện tại stratify theo ảnh, không tách riêng người chụp. Có 432 user_id chung giữa train/validation, 446 giữa train/test và 219 giữa validation/test. Điểm test có thể hưởng lợi từ việc cùng người xuất hiện ở nhiều tập; **chưa đại diện hoàn toàn cho nhận diện người mới**.

Ảnh test được crop bằng bounding box có sẵn. Webcam dùng ROI thủ công và điều kiện ánh sáng/nền khác dataset, nên accuracy test 94,58% **không phải accuracy webcam**. Cần kiểm thử riêng webcam với đủ 8 cử chỉ.

## 8. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError` | Dùng đúng Python trong `.venv`, cài lại `requirements.txt` nếu thiếu thư viện |
| Thiếu cache/checkpoint/model | Chạy lần lượt bước 6 → 7 → 8; không cần làm lại nếu artifact đã có và còn khớp |
| Pipeline/source không khớp hash | Nếu đã sửa tiền xử lý/HOG, chạy lại `--prepare` → train → đánh giá để artifact đồng bộ |
| Matplotlib lỗi Tcl/Tk hoặc không mở được cửa sổ | Thêm `--no-show` cho demo ảnh, preprocessing hoặc feature_extraction; xem PNG đã lưu |
| Webcam không mở được | Đóng ứng dụng đang dùng camera, kiểm tra quyền camera của Windows, thử `--camera 1` |
| Chọn ROI xong không thấy kết quả | Dùng bản code hiện tại, xem cửa sổ webcam; nhấn Space nếu đang tạm dừng |
| ROI có mặt hoặc nền nhưng vẫn hiện cử chỉ | Chọn lại đúng bàn tay; model chưa có nhãn “không có tay” |
| Dự đoán webcam nhảy hoặc sai | Khoanh sát tay, giữ tay trong khung, đủ sáng, tránh nền phức tạp; ROI chưa tự theo dõi |
| Thông báo pip có bản mới | Chỉ là thông báo; không cần cập nhật để chạy project hiện tại |

Khi cần tái lập kết quả, giữ cùng dataset, seed, source xử lý và phiên bản thư viện. Không cần cài TensorFlow hay thay đổi `.idea/`.
