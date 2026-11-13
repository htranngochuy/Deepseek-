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



### PowerShell

```powershell
gcc -std=c11 -Wall -Wextra -Iinclude test/main.c source/data_type.c source/load.c source/kernel.c -lm -o test/app.exe
```
Sau khi hiện ra được file app.exe thì chạy lệnh .\test\app.exe trong PS 

