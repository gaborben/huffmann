CC = gcc
CXX = g++
CFLAGS = -Iinclude
CXXFLAGS = -Iinclude -IC:/msys64/mingw64/include
LDFLAGS = -LC:/msys64/mingw64/lib -lgtest -lgtest_main -pthread

SRC = src/kernel_loader.c src/huffman.c
MAIN = main.c
TARGET = main.exe

TEST_SRC = tests/test_huffman.cpp
TEST_BIN = test_runner.exe

all:
	$(CC) $(MAIN) $(SRC) -o $(TARGET) $(CFLAGS) -lOpenCL

test:
	$(CXX) $(TEST_SRC) $(SRC) -o $(TEST_BIN) $(CXXFLAGS) $(LDFLAGS)

clean:
	del /Q $(TARGET) $(TEST_BIN) 2>nul || exit 0
