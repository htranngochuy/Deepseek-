CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Iinclude
LDLIBS  = -lm
SRCS    = test/main.c $(wildcard source/*.c)
HDRS    = $(wildcard include/*.h)
TARGET  = test/app.exe

# make khong co sh (cmd.exe thuan) thi dung del, con lai dung rm
ifneq ($(and $(filter Windows_NT,$(OS)),$(findstring cmd,$(SHELL))),)
  CLEAN = del /Q /F $(subst /,\,$(TARGET)) 2>nul
else
  CLEAN = rm -f $(TARGET)
endif

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(SRCS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	-$(CLEAN)

.PHONY: all run clean