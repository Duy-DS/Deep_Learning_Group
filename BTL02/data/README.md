# Hướng Dẫn Quản Lý Dữ Liệu FER2013 (Data Management Guide)

Thư mục này quản lý toàn bộ dữ liệu phục vụ cho bài toán **Nhận diện cảm xúc khuôn mặt (Facial Emotion Recognition - FER2013)**.

---

## 1. Cấu trúc thư mục dữ liệu

```text
data/
├── raw/                      # Dữ liệu gốc tải về từ Kaggle
│   └── fer2013.csv           # File CSV gốc gồm 35,887 dòng pixel 48x48
├── processed/                # Dữ liệu đã tiền xử lý & phân chia tập
│   ├── train/                # Tập huấn luyện (Training: ~28,709 mẫu)
│   │   ├── 0_angry/
│   │   ├── 1_disgust/
│   │   ├── 2_fear/
│   │   ├── 3_happy/
│   │   ├── 4_sad/
│   │   ├── 5_surprise/
│   │   └── 6_neutral/
│   ├── public_test/          # Tập kiểm định (Validation / PublicTest: 3,589 mẫu)
│   └── private_test/         # Tập kiểm thử cuối kỳ (Test / PrivateTest: 3,589 mẫu)
└── README.md                 # Tài liệu này
```

---

## 2. Thông tin về tập dữ liệu FER2013

- **Định dạng ảnh:** Ảnh xám (Grayscale), kích thước $48 \times 48$ pixels.
- **Tổng số mẫu:** 35,887 ảnh khuôn mặt đã được căn giữa.
- **Các lớp cảm xúc (7 Emotion Classes):**
  - `0`: **Angry** (Tức giận)
  - `1`: **Disgust** (Ghê tởm)
  - `2`: **Fear** (Sợ hãi)
  - `3`: **Happy** (Hạnh phúc / Vui vẻ)
  - `4`: **Sad** (Buồn rầu)
  - `5`: **Surprise** (Ngạc nhiên)
  - `6`: **Neutral** (Trung tính)

- **Phân chia mặc định từ FER2013:**
  - `Training`: 28,709 mẫu (~80%)
  - `PublicTest`: 3,589 mẫu (~10%)
  - `PrivateTest`: 3,589 mẫu (~10%)

---

## 3. Hướng dẫn tải dữ liệu

Nguồn dữ liệu chuẩn duy nhất hiện tại là cuộc thi Kaggle: [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

1. **Cấp quyền Kaggle:** đăng nhập, nhấn **Accept Rules** tại trang dữ liệu cuộc thi, tạo API Token trong Kaggle Account Settings và đặt `kaggle.json` tại `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`). Có thể dùng biến môi trường `KAGGLE_USERNAME` và `KAGGLE_KEY` thay thế.
2. **Tải và xác thực bằng một lệnh** từ thư mục `BTL02/`:
   ```bash
   bash scripts/download_data.sh
   ```
   Trên Windows có thể chạy `python -m src.data.downloader`. Tùy chọn `--force` tải lại; `--verify-only` chỉ xác thực file hiện có.
3. **Kiểm tra kết quả:** chương trình xác nhận ba cột `emotion`, `pixels`, `Usage`, đúng 35.887 dòng và phân chia 28.709 / 3.589 / 3.589; cuối cùng in SHA-256 để đối chiếu.

File chuẩn sau khi chạy là `data/raw/fer2013.csv`. Dữ liệu chỉ được thay thế sau khi archive đã giải nén và CSV ứng viên vượt qua toàn bộ kiểm tra. Google Drive dự phòng hiện đang cập nhật; dùng Kaggle API làm nguồn chuẩn.

---

## 4. Quy ước MLOps & Quản lý Git

- **Tuyệt đối không commit dữ liệu lớn vào Git:** Thư mục `data/raw/*` và `data/processed/*` đã được cấu hình trong `.gitignore` để tránh phình dung lượng kho lưu trữ.
- **Gitkeep:** Các thư mục rỗng được theo dõi thông qua file `.gitkeep`.
- **DVC / Artifact Store:** Nếu triển khai pipeline lớn, cấu hình DVC hoặc lưu trữ trên Google Drive / Cloud Storage và chỉ ghi đường dẫn tại `configs/paths.yaml`.

