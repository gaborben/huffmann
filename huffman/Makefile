CC       = gcc
CFLAGS   = -Iinclude
LDFLAGS  = -lOpenCL

SRC      = src/kernel_loader.c src/huffman.c
MAIN     = src/main.c
BUILD_DIR= build
TARGET   = $(BUILD_DIR)/main.exe

.PHONY: all clean

all:
	@cls
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CC) $(MAIN) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS)
	@$(TARGET)

clean:
	@cls
	@if exist $(TARGET) del /Q $(TARGET)
