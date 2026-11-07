# AGENTS.md

Hướng dẫn đầy đủ cho mọi coding agent nằm trong **`CLAUDE.md`** (cùng thư mục). Đọc file đó trước khi làm bất cứ việc gì.

Tóm tắt bốn quy tắc cốt lõi:
1. Viết và sửa code ở máy local.
2. Mọi việc cần GPU chạy trên Kaggle qua CLI (`python kaggle/kaggle_cli.py ...`), việc nhẹ chạy ở local.
3. Mọi lần chạy phải có file log tương ứng (`logs/` ở local, `kaggle_out/<slug>/<run_id>/` cho Kaggle).
4. Không đưa khóa vào code hay log; không xóa gì trên Kaggle khi chưa được yêu cầu rõ.
