# Flatten + RMSNorm (khối Hyperconnection)

Mô hình C của khối: `DDR4 (bf16) -> RAB -> thanh ghi -> Flatten (fp32) -> RMSNorm (fp32)`.

## Cấu trúc thư mục

```
DA1/
├── include/
│   ├── config.h           hằng số (số token, N_HC, D_MODEL, ...)
│   ├── data_type.h        kiểu bf16 và chuyển đổi bf16 <-> fp32
│   ├── data_structure.h   struct DDR4_format, RAB_format, registers_format
│   ├── load.h             load_gl_x_i (DDR4 -> RAB), load_lc_x_i (RAB -> thanh ghi)
│   └── kernel.h           flatten, rmsnorm
├── source/
│   ├── data_type.c
│   ├── load.c
│   └── kernel.c
└── test/
    └── main.c             chương trình chạy thử
```

## Cách chạy (không cần make)

Chỉ cần có `gcc`. Mở terminal **ở thư mục gốc `DA1`** (thư mục chứa `include/`, `source/`, `test/`).

### Windows (PowerShell)

```powershell
gcc -O2 -Wall -Wextra -Iinclude -o main test/main.c source/data_type.c source/kernel.c source/load.c -lm
.\main
```

### Linux / macOS

```bash
gcc -O2 -Wall -Wextra -Iinclude -o main test/main.c source/data_type.c source/kernel.c source/load.c -lm
./main
```

Ý nghĩa các cờ:

| Cờ | Tác dụng |
|---|---|
| `-Iinclude` | chỉ cho compiler tìm file `.h` trong thư mục `include/` |
| `-lm` | liên kết thư viện toán (cần cho `sqrt`) |
| `-Wall -Wextra` | bật cảnh báo |
| `-o main` | đặt tên file chạy là `main` (`main.exe` trên Windows) |

Phải liệt kê từng file `.c` trong `source/`. Nếu thêm file `.c` mới thì thêm tên nó vào lệnh.

## Kết quả

Chương trình chạy xong sẽ in các giá trị `y_i` (đầu ra RMSNorm, fp32) của token cuối cùng rồi thoát bình thường.

## Cài gcc nếu chưa có

- **Windows:** cài MSYS2 hoặc MinGW-w64, rồi thêm thư mục `bin` của nó vào biến môi trường `PATH`. Kiểm tra bằng `gcc --version`.
- **Ubuntu/Debian:** `sudo apt install build-essential`
- **macOS:** `xcode-select --install`

## Lỗi thường gặp

| Thông báo | Nguyên nhân và cách sửa |
|---|---|
| `gcc is not recognized` | chưa cài gcc hoặc chưa thêm vào `PATH` |
| `'xxx.h' file not found` | chưa chạy ở thư mục gốc `DA1`, hoặc thiếu `-Iinclude` |
| `undefined reference to ...` | thiếu một file `.c` trong lệnh biên dịch |
| `undefined reference to 'sqrt'` | thiếu `-lm` ở cuối lệnh |

## Dùng make (tuỳ chọn)

Nếu có `Makefile` ở thư mục gốc và máy có `make`:

```
make run                # Windows (MinGW): mingw32-make run
```

Nếu `make` không chạy được thì dùng lệnh `gcc` ở trên, kết quả giống nhau.

## Ghi chú

- `RMS.c` ở thư mục gốc là bản cũ, không nằm trong lệnh biên dịch ở trên.
- `gamma` hiện là một số vô hướng, chưa phải vector.
