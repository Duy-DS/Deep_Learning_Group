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

1. Truy cập cuộc thi Kaggle FER2013: [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. Tải file `fer2013.tar.gz` hoặc `fer2013.csv`.
3. Giải nén và đặt file `fer2013.csv` vào thư mục `data/raw/`.
4. Chạy script tiền xử lý và giải nén thành định dạng thư mục ảnh:
   ```bash
   python scripts/prepare_data.py --input data/raw/fer2013.csv --output data/processed/
   ```

---

## 4. Quy ước MLOps & Quản lý Git

- **Tuyệt đối không commit dữ liệu lớn vào Git:** Thư mục `data/raw/*` và `data/processed/*` đã được cấu hình trong `.gitignore` để tránh phình dung lượng kho lưu trữ.
- **Gitkeep:** Các thư mục rỗng được theo dõi thông qua file `.gitkeep`.
- **DVC / Artifact Store:** Nếu triển khai pipeline lớn, cấu hình DVC hoặc lưu trữ trên Google Drive / Cloud Storage và chỉ ghi đường dẫn tại `configs/paths.yaml`.

