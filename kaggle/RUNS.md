# Sổ ghi các lần chạy Kaggle

Đọc nhanh một lần chạy: `python kaggle/digest.py <slug> [--run RUN_ID]`.

| # | Giờ local | Slug / thư mục log | GPU | Kết quả |
|---|---|---|---|---|
| 1 | 2026-11-07 21:10 | `brp-smoke-l2br` / `kaggle_out/brp-smoke-l2br/20261107_211000_pathbug` | Tesla T4, cuda True | **Lỗi** `FileNotFoundError: datasets\CVS`: loader L2BR dùng đường dẫn Windows. GPU hoạt động, không có điểm số |
| 2 | 2026-11-07 21:12 | `brp-smoke-l2br` / `kaggle_out/brp-smoke-l2br/20261107_211200` | Tesla T4, cuda True | **Đạt.** Tái lập Bảng 2 (L2BR Base, CVS uBRP, H=3, N=2560): S=3 → 4.975, S=4 → 6.025, S=5 → 6.850, đúng bằng bài báo (chênh 0,0%). Mỗi nhóm 15–20 giây. Đã vá đường dẫn bằng thay chuỗi trong `main.py` |
| 3 | 2026-11-07 21:16 | `brp-g0-headroom-cvs10x6` / `kaggle_out/brp-g0-headroom-cvs10x6/20261107_211651` | Tesla T4 | **Đạt cổng G0.** CVS uBRP H=10 S=6, 40 bài, T=1, mô hình SIL công khai: M=256 → 66,150 (55 s, 1,4 s/bài); M=2560 → 65,275 (262 s, 6,5 s/bài); M=25600 → 64,700 (2381 s, 59,5 s/bài). Giảm 0,881% từ M=2560 lên 25600 (ngưỡng 0,3%): còn dư địa. Bài báo: L2BR-Large 66,70, Targeted 65,20 (chọn nhiệt độ tốt nhất) |
| 4 | 2026-11-07 21:25 | `brp-resume-selftest` / `kaggle_out/brp-resume-selftest/{20261107_212454,20261107_212631,20261107_212842}` | CPU | **Đạt.** 3 phiên tự nối tiếp: 4 → 8 → 9 mục, không tính lại, phiên cuối `finished: true`. Code lấy từ GitHub commit `88ac2e1` |

Ghi chú: `status.log` và `live.log` của lần 2 còn lẫn các dòng của lần 1, vì lần 2 chạy bằng phiên bản `kaggle_cli.py` cũ (chưa tách thư mục mỗi lần chạy).
