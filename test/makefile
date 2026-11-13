# Dat file nay o thu muc goc DA1 (cung cap voi include/, source/, test/)
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -Iinclude
LDLIBS  = -lm
SRCS    = test/main.c $(wildcard source/*.c)
HDRS    = $(wildcard include/*.h)

ifeq ($(OS),Windows_NT)
  TARGET = main.exe
  RUN    = $(TARGET)
  CLEAN  = del /Q /F $(TARGET) 2>nul
else
  TARGET = main
  RUN    = ./$(TARGET)
  CLEAN  = rm -f $(TARGET)
endif

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)

run: $(TARGET)
	$(RUN)

clean:
	-$(CLEAN)

.PHONY: all run clean