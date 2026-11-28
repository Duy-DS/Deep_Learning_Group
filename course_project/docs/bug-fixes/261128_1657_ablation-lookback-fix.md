# Biên bản Sửa lỗi: Thiếu tham số `--lookback` trong `run_ablation.py` & Gia cố Backup Cloud (`BPM25_T0305_ABLATION_FIX`)

- **Ngày thực hiện:** 2026-11-28
- **Người thực hiện:** Đinh Thái Toàn
- **Nhánh Git:** `fix/BPM25_T0305_wp03-training-audit`
- **Tệp liên quan:** `run_ablation.py`, `src/utils/exp_backup.py`, `notebooks/kaggle/train_and_finetune_kaggle.ipynb`, `notebooks/colab/train_and_finetune_colab.ipynb`

---

## 1. Triệu chứng lỗi & Raw Traceback
Khi chạy thực nghiệm trên Kaggle sau khi hoàn tất phần Fine-tuning (`before_vs_after_by_model_multivariate_H192.csv`), tiến trình sang bước Ablation thì dừng đột ngột tại `run_ablation.py:581`:

```text
Traceback (most recent call last):
  File "run_ablation.py", line 852, in <module>
    run_ablation_suite(parse_args())
  File "run_ablation.py", line 645, in run_ablation_suite
    multi_h24 = _get_pretrain("multivariate", 24, DEFAULT_MODELS)
  File "run_ablation.py", line 581, in _get_pretrain
    lookback=args.lookback,
AttributeError: 'Namespace' object has no attribute 'lookback'
```

## 2. Nguyên nhân gốc rễ (Root Cause)
- Hàm `parse_args()` trong `run_ablation.py` chưa khai báo tùy chọn `--lookback`.
- Hàm `apply_yaml_config(parser, ["paths", "data", "pretrain", "ablation"])` chỉ nạp giá trị từ `configs/default.yaml` cho các tham số đã được đăng ký sẵn trên `ArgumentParser`, nên `data.lookback` bị bỏ qua và `args.lookback` không tồn tại trên đối tượng `Namespace` khi được truy xuất ở dòng 581, 617 và 699.
- Ngoài ra, trên Kaggle/Colab, nếu chạy rời rạc từng cell hoặc phiên bị ngắt do timeout, cần cố định thư mục làm việc (`CWD`) về `course_project/` ở đầu mỗi cell và tự động chạy backup (`--upload-gdrive`) ngay cuối cell huấn luyện chính.

## 3. Đoạn mã sửa đổi (Code Diff)

### 3.1. Bổ sung `--lookback` vào `run_ablation.py`
```diff
     parser.add_argument("--num-stations", type=int, default=None)
+    parser.add_argument(
+        "--lookback",
+        type=int,
+        default=96,
+        help="Chiều dài cửa sổ lịch sử L (mặc định: 96 hoặc đọc từ data.lookback trong YAML).",
+    )
     parser.add_argument("--batch-size", type=int, default=512)
```
*(Đã rà soát toàn bộ 34 thuộc tính `args.*` được truy xuất trong `run_ablation.py`; tất cả đều đã tồn tại trong `parse_args()`).*

### 3.2. Bổ sung `--upload-gdrive` vào `src/utils/exp_backup.py` & Gia cố 2 Notebook
- Bổ sung phương thức `ExperimentBackupManager.upload_to_gdrive()` và cờ CLI `--upload-gdrive` trong `src/utils/exp_backup.py` (tự động nhận diện `google.colab.auth`, biến môi trường `.env` hoặc `kaggle_secrets.UserSecretsClient`).
- Cố định `CWD` về `course_project/` ở đầu mỗi cell chạy script trong `notebooks/kaggle/train_and_finetune_kaggle.ipynb` và `notebooks/colab/train_and_finetune_colab.ipynb`.
- Gắn `!python -m src.utils.exp_backup --include-checkpoints --upload-gdrive` ngay cuối cell huấn luyện & ablation chính, đồng thời sao chép `.zip` và `mlflow_runs_export.csv` ra `/kaggle/working/` trên Kaggle để lưu vào tab **Output** khi dùng **Save Version (Save & Run All)**.

## 4. Bằng chứng kiểm thử sau sửa đổi
Lệnh kiểm thử:
1. `python -m py_compile run_ablation.py src/utils/exp_backup.py` -> Exit code `0`
2. `python run_ablation.py --help` -> Hiển thị đầy đủ `--lookback LOOKBACK`
3. Smoke test `run_ablation.parse_args()` với `--config configs/default.yaml`:
   ```text
   SMOKE TEST OK: lookback= 96 all 34 attrs present
   JSON OK: notebooks/kaggle/train_and_finetune_kaggle.ipynb 9 cells
   JSON OK: notebooks/colab/train_and_finetune_colab.ipynb 13 cells
   ```
