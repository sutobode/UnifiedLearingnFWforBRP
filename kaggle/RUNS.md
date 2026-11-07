# Sổ ghi các lần chạy Kaggle

Đọc nhanh một lần chạy: `python kaggle/digest.py <slug> [--run RUN_ID]`.

| # | Giờ local | Slug / thư mục log | GPU | Kết quả |
|---|---|---|---|---|
| 1 | 2026-11-07 21:10 | `brp-smoke-l2br` / `kaggle_out/brp-smoke-l2br/20261107_211000_pathbug` | Tesla T4, cuda True | **Lỗi** `FileNotFoundError: datasets\CVS`: loader L2BR dùng đường dẫn Windows. GPU hoạt động, không có điểm số |
| 2 | 2026-11-07 21:12 | `brp-smoke-l2br` / `kaggle_out/brp-smoke-l2br/20261107_211200` | Tesla T4, cuda True | **Đạt.** Tái lập Bảng 2 (L2BR Base, CVS uBRP, H=3, N=2560): S=3 → 4.975, S=4 → 6.025, S=5 → 6.850, đúng bằng bài báo (chênh 0,0%). Mỗi nhóm 15–20 giây. Đã vá đường dẫn bằng thay chuỗi trong `main.py` |

Ghi chú: `status.log` và `live.log` của lần 2 còn lẫn các dòng của lần 1, vì lần 2 chạy bằng phiên bản `kaggle_cli.py` cũ (chưa tách thư mục mỗi lần chạy). Từ lần 3 trở đi mỗi lần chạy có thư mục riêng.
