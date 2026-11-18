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

1. **Chuẩn bị cấu hình cá nhân** từ thư mục `BTL02/`: cài `requirements.txt`, chạy `cp .env.example .env` (PowerShell: `Copy-Item .env.example .env`). Đăng nhập Kaggle, nhấn **Accept Rules** tại link gốc và tạo API Access Token. Điền token mới vào `KAGGLE_API_TOKEN` trong `.env`; không dùng lại token đã công khai và không commit `.env`.
2. **Tải và xác thực bằng một lệnh** từ thư mục `BTL02/`:
   ```bash
   bash scripts/download_data.sh
   ```
   Trên Windows có thể chạy `python -m src.data.downloader`. Tùy chọn `--force` tải lại; `--verify-only` chỉ xác thực file hiện có.
3. **Kiểm tra kết quả:** chương trình tự nạp `.env`, xác nhận ba cột `emotion`, `pixels`, `Usage`, đúng 35.887 dòng và phân chia 28.709 / 3.589 / 3.589; cuối cùng in SHA-256 để đối chiếu.

| Phân tập | Số mẫu chuẩn |
| :--- | ---: |
| Training | 28.709 |
| PublicTest | 3.589 |
| PrivateTest | 3.589 |
| Tổng cộng | 35.887 |

Downloader tìm `.env` tại gốc BTL02 trước, nếu không có mới xét thư mục làm việc hiện tại; không ghi đè biến môi trường đã có và không nội suy giá trị token. Không tự tìm `.env` ở các thư mục cha khác. Có thể dùng `~/.kaggle/access_token` thay `.env`. Thứ tự chứng thực: `KAGGLE_API_TOKEN` (shell hoặc `.env`) → `access_token` → legacy env/json. Chỉ `.env.example` chứa placeholder được đưa vào Git.

File chuẩn sau khi chạy là `data/raw/fer2013.csv`. Downloader gọi `kagglehub.competition_download` và không xóa/sửa cache chung của kagglehub. CSV chỉ được thay thế nguyên tử sau khi archive đã giải nén và ứng viên vượt qua kiểm tra header, split, nhãn 0–6 và 2304 pixel [0,255]/ảnh. SHA-256 dùng để đối chiếu byte giữa các thành viên, không phải chữ ký xác thực từ Kaggle. Google Drive dự phòng **chưa tồn tại**; dùng trực tiếp Kaggle làm nguồn chuẩn duy nhất.

Ví dụ cấu hình token (thay giá trị minh họa tại máy cá nhân, không lưu vào repo):
```bash
export KAGGLE_API_TOKEN='<YOUR_TOKEN>'
bash scripts/download_data.sh
```
PowerShell: `$env:KAGGLE_API_TOKEN='<YOUR_TOKEN>'`, sau đó `python -m src.data.downloader` từ thư mục BTL02. Cài `requirements.txt` trước khi sử dụng.

---

## 4. Quy ước MLOps & Quản lý Git

- **Tuyệt đối không commit dữ liệu lớn vào Git:** Thư mục `data/raw/*` và `data/processed/*` đã được cấu hình trong `.gitignore` để tránh phình dung lượng kho lưu trữ.
- **Gitkeep:** Các thư mục rỗng được theo dõi thông qua file `.gitkeep`.
- **DVC / Artifact Store:** Nếu triển khai pipeline lớn, cấu hình DVC hoặc lưu trữ trên Google Drive / Cloud Storage và chỉ ghi đường dẫn tại `configs/paths.yaml`.

