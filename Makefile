# NEAT (NeuroEvolution of Augmenting Topologies) Implementation
# Makefile for building the project

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -fopenmp -msse4.2 -mavx2 -mfma -std=c11
DEBUG_CFLAGS = -g -O0 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
LDFLAGS = -lm -lSDL2 -lSDL2_ttf -fopenmp

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
EXAMPLES_DIR = examples
TESTS_DIR = tests
BENCHMARKS_DIR = benchmarks

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Example programs
EXAMPLES = visualization_demo hyperneat_demo novelty_search_demo interactive_editor
EXAMPLE_BINS = $(addprefix $(BIN_DIR)/, $(EXAMPLES))

# Test files
TEST_SRCS = $(wildcard $(TESTS_DIR)/*.c)
TEST_BINS = $(patsubst $(TESTS_DIR)/%.c,$(BIN_DIR)/%,$(TEST_SRCS))

# Benchmark files
BENCHMARK_SRCS = $(wildcard $(BENCHMARKS_DIR)/*.c)
BENCHMARK_BINS = $(patsubst $(BENCHMARKS_DIR)/%.c,$(BIN_DIR)/%,$(BENCHMARK_SRCS))

# Main targets
.PHONY: all debug release clean test examples benchmarks

# Default build (release mode)
all: release

# Debug build with sanitizers and debug symbols
debug: CFLAGS += $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: clean $(BIN_DIR)/libneat.a $(BIN_DIR)/libneat.so $(BIN_DIR)/neat_test_runner

# Release build with optimizations
release: CFLAGS += -DNDEBUG
release: clean $(BIN_DIR)/libneat.a $(BIN_DIR)/libneat.so $(BIN_DIR)/neat_test_runner

# Create necessary directories
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Build static library
$(BIN_DIR)/libneat.a: $(OBJS) | $(BIN_DIR)
	ar rcs $@ $^

# Build shared library
$(BIN_DIR)/libneat.so: $(OBJS) | $(BIN_DIR)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# Build test runner
$(BIN_DIR)/neat_test_runner: $(TESTS_DIR)/test_runner.c $(BIN_DIR)/libneat.a | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(BIN_DIR) -lneat $(LDFLAGS)

# Build individual test binaries
$(BIN_DIR)/%: $(TESTS_DIR)/%.c $(BIN_DIR)/libneat.a | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(BIN_DIR) -lneat $(LDFLAGS)

# Build example programs
$(BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(BIN_DIR)/libneat.a | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(BIN_DIR) -lneat $(LDFLAGS) `sdl2-config --cflags --libs` -lSDL2_ttf

# Build benchmark programs
$(BIN_DIR)/%: $(BENCHMARKS_DIR)/%.c $(BIN_DIR)/libneat.a | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(BIN_DIR) -lneat $(LDFLAGS)

# Build all examples
examples: $(BIN_DIR)/libneat.a $(addprefix $(BIN_DIR)/, $(EXAMPLES))

# Build all tests
tests: $(BIN_DIR)/libneat.a $(BIN_DIR)/neat_test_runner

# Build all benchmarks
benchmarks: $(BIN_DIR)/libneat.a $(BENCHMARK_BINS)

# Run tests
test: tests
	@echo "Running tests..."
	@$(BIN_DIR)/neat_test_runner

# Run benchmarks
bench: benchmarks
	@echo "Running benchmarks..."
	@for bench in $(BENCHMARK_BINS); do \
		echo "\n=== Running $$(basename $$bench) ==="; \
		$$bench; \
	done

# Clean build artifacts
clean:
	@echo "Cleaning build..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install dependencies (Linux/Ubuntu)
install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install -y build-essential libsdl2-dev libsdl2-ttf-dev

# Generate documentation (requires Doxygen)
docs:
	@doxygen Doxyfile

# Run with Valgrind for memory checking
memcheck: debug
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(BIN_DIR)/neat_test_runner

# Format code (requires clang-format)
format:
	find $(SRC_DIR) $(INCLUDE_DIR) $(TESTS_DIR) $(EXAMPLES_DIR) -name "*.c" -o -name "*.h" | xargs clang-format -i

# Static analysis (requires cppcheck)
analyze:
	cppcheck --enable=all --inconclusive -I $(INCLUDE_DIR) $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

# Package the library for distribution
package: release
	@echo "Creating distribution package..."
	@mkdir -p dist/neat/include dist/neat/lib
	@cp -r $(INCLUDE_DIR)/*.h dist/neat/include/
	@cp $(BIN_DIR)/libneat.a $(BIN_DIR)/libneat.so dist/neat/lib/
	@cp README.md LICENSE dist/neat/
	@tar -czvf neat-$(shell date +%Y%m%d).tar.gz -C dist .
	@rm -rf dist
	@echo "Package created: neat-$(shell date +%Y%m%d).tar.gz"

# Help target
help:
	@echo "NEAT (NeuroEvolution of Augmenting Topologies) Build System"
	@echo "--------------------------------------------------"
	@echo "Available targets:"
	@echo "  all/release: Build the project in release mode (default)"
	@echo "  debug:        Build with debug symbols and sanitizers"
	@echo "  clean:        Remove all build artifacts"
	@echo "  test:         Build and run all tests"
	@echo "  examples:     Build example programs"
	@echo "  benchmarks:   Build benchmark programs"
	@echo "  bench:        Run all benchmarks"
	@echo "  docs:         Generate documentation (requires Doxygen)"
	@echo "  format:       Format source code (requires clang-format)"
	@echo "  analyze:      Run static analysis (requires cppcheck)"
	@echo "  package:      Create a distributable package"
	@echo "  help:         Show this help message"

# Default target
.DEFAULT_GOAL := all
