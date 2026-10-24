# Nhận diện cử chỉ tay

Chương trình sử dụng HOG và SVM để nhận diện 8 cử chỉ: `fist`, `palm`, `like`, `dislike`, `peace`, `ok`, `one`, `stop`.

## Chuẩn bị

Mở terminal PowerShell tại thư mục project, sử dụng môi trường `.venv` đã cài các thư viện trong `requirements.txt`.

Tải **[models.zip tại Release v1.0.0](https://github.com/Mika204/NhanDangCuChiTay/releases/tag/v1.0.0)** rồi giải nén tại thư mục gốc project. Sau khi giải nén phải có:

```text
models/
├── gesture_model.pkl
├── scaler.pkl
├── class_names.pkl
└── model_metadata.json
```

Model nằm trong Release thay vì Git vì file model khoảng 230 MB. Dataset và cache HOG không được đưa lên repo.

## 1. Chạy webcam

```powershell
.\.venv\Scripts\python.exe .\main.py webcam
```

1. Đưa bàn tay lên trước camera.
2. Nhấn **R**, kéo chuột khoanh sát toàn bộ bàn tay.
3. Nhấn **Enter** để xác nhận và tự bắt đầu dự đoán.
4. Xem **Gesture** và **Confidence** trên cửa sổ webcam.

**Space** tạm dừng/chạy tiếp; **R** chọn lại vùng tay; **Q** hoặc **Esc** thoát.

Khung không tự di chuyển theo tay. Giữ tay trong khung, đủ sáng và hạn chế nền. Chương trình chưa tự phát hiện trường hợp không có tay. **Webcam chỉ cần model đã tải, không cần dataset hoặc cache.**

Nếu camera mặc định không mở được:

```powershell
.\.venv\Scripts\python.exe .\main.py webcam --camera 1
```

## 2. Hiển thị và lưu ảnh dự đoán

```powershell
.\.venv\Scripts\python.exe .\test_model.py
```

Mỗi lần chạy tự chọn ngẫu nhiên 8 ảnh test, hiển thị nhãn thật, nhãn dự đoán và độ tin cậy. Không trùng ảnh trong một lần chọn; giữa các lần vẫn có thể gặp lại ảnh cũ.

Hình lưu tại **`outputs/figures/test_predictions.png`**, ghi đè khi chạy lại. Kết quả từng ảnh lưu tại `outputs/results/step9_predictions.csv`.

**Chức năng test ảnh cần thêm dữ liệu local:** dataset gốc trong `ann_train_val/` và `hagrid_30k/`, `outputs/results/hog_features.npz` và `outputs/results/split_indices.npz`. Gói models.zip không chứa các dữ liệu này. Nếu chỉ tải repo và model, có thể chạy webcam nhưng chưa chạy được test ảnh.

Nếu chỉ muốn lưu hình, không mở cửa sổ:

```powershell
.\.venv\Scripts\python.exe .\test_model.py --no-show
```

Các lệnh trên dùng model đã lưu, không huấn luyện lại.

## Hình kết quả

- [Confusion matrix](outputs/figures/confusion_matrix.png)
- [So sánh KNN và SVM](outputs/figures/knn_vs_svm.png)
- [Ví dụ dự đoán 8 ảnh test](outputs/figures/test_predictions.png)

SVM đạt accuracy **94,58%** trên 960 ảnh test. Các tập còn trùng người chụp; đây không phải accuracy webcam hay đánh giá hoàn toàn trên người mới.

