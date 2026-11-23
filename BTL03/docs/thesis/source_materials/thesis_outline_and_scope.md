# Đề cương và phạm vi snapshot báo cáo

## Snapshot

- Dự án: `BTL03/`.
- Commit nguồn được xét: `33dcd74599e86404bbc8128c8057afba82810093`, báo cáo T0212 tại `docs/reports/T0212_261123_2130.md`.
- Thời điểm evidence T0212 ghi: 2026-11-23 21:30; không coi mốc thời gian/tên file là bằng chứng của run khác.
- Mục tiêu báo cáo: phản ánh quá trình chuyển mười cấu hình thuộc năm nhóm từ Keras sang PyTorch, gồm implementation, parameter-count audit, kiểm thử và giới hạn; phân biệt kết quả đã đo với kế hoạch.

## Phạm vi nội dung

1. Thông tin đề tài/nhóm; thành viên dùng alias đã có chứng cứ. Họ tên pháp lý, MSSV, chủ trì cuối kỳ và tỷ lệ phải được nhóm xác nhận.
2. Nền tảng chuyển đổi: NHWC→NCHW, xoay WIKI `np.rot90(k=3)` trước permute, `TimeDistributed`→reshape CNN/LSTM, upsampling/SegNet, score-map FCN, chuẩn hóa và split.
3. Kiến trúc thực tế trong repo: `BaseMigratedModel`, `BaseTrainer`, seed helpers, callbacks/logger/checkpoint, YAML và CLI. Phân biệt contract code với lần chạy thành công.
4. Snapshot T0212: PR/history audit, mười parameter counts được report, 10 smoke synthetic one-epoch cases, logger blockers và kết quả suite.
5. Trạng thái theo task; ghi nhận T0201 sweep 36 run và T0202 MNIST MLP 30 epoch có evidence, nhưng không mở rộng thành kết quả full training của mười mô hình.
6. WP03 và WP04 chưa nghiệm thu tại snapshot theo report/task workbook: ghi `[CHƯA THỰC HIỆN - DỰ KIẾN Ở GIAI ĐOẠN TIẾP THEO]` cho hạng mục chưa chạy/hoàn tất; tách việc `BLOCKED` và code đã triển khai.
7. Hạn chế: thiếu dữ liệu WIKI/FER/SegNet/vehicle trong workspace; CPU-only; dependency TensorBoard/protobuf; protocol legacy validation/public-test; thiếu baseline numeric ở một số trường; README không phải nguồn ưu tiên khi mâu thuẫn.
8. Giữ nguyên bất đồng parameter count CNN-LSTM giữa report T0206 (1,746,049) và T0207/T0212 (1,747,073); đối chiếu source/commit/counting semantics trước khi chốt parity.

## Không thuộc phạm vi

- Không tuyên bố nghiệm thu toàn bộ T0212: report gốc ghi `BLOCKED`, logger 10/10 bị chặn, suite 78 passed/4 failed trên 82 test.
- Không tạo accuracy PyTorch, IoU/mAP, thời gian train hay parity metric không có nguồn.
- Không gọi smoke synthetic là huấn luyện bằng dataset thật.
- Không tạo chart/prediction hình ảnh giả để lấp chỗ trống.
- Không sửa model source, `legacy_keras/`, dữ liệu, requirements, `main` hoặc lịch sử commit.
- Không đưa PDF giả/bản build tạm vào repository.

## Quy tắc phiên bản sau T0212

Mỗi đợt cập nhật phải ghi commit/hash, thời điểm, evidence manifest và trạng thái tại mốc mới. Không hồi tố hoặc ghi đè snapshot T0212. Kết quả mới cần nguồn có run ID, seed, config, split, runtime, metric definition, log/checkpoint và report nghiệm thu. Chỉ trình bày kết luận phù hợp với thiết kế đánh giá đã xác nhận.
