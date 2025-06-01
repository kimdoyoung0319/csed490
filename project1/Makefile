.PHONY: all clean runp runs check view

BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
TARGET_DIR := $(BUILD_DIR)/target
SRC_DIR := src

CXX := g++
CXXFLAGS := -Wall -O2 -fopenmp -Wno-unknown-pragmas
LDFLAGS := -lrt
DEBUG_FLAGS := -Wall -O0 -g -fopenmp -Wno-unknown-pragmas
SERIAL_FLAGS := -Wall -O2 -g -Wno-unknown-pragmas

TARGETS := lu lu-serial lu-debug

SRC_FILES := $(SRC_DIR)/lu.cpp $(SRC_DIR)/decomp.cpp $(SRC_DIR)/matrix.cpp
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
OBJ_FILES_DEBUG := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/debug/%.o, $(SRC_FILES))
OBJ_FILES_SERIAL := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/serial/%.o, $(SRC_FILES))

MATRIX_SIZE := 8000
MATRIX_CHECK_SIZE := 100
NUM_CORES := `grep processor /proc/cpuinfo | wc -l`

CHECKER := inspxe-cl -collect=ti3 -r check
VIEWER := inspxe-gui check

all: $(TARGETS)

lu: $(OBJ_FILES) | $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET_DIR)/$@ $^ $(LDFLAGS) 

lu-debug: $(OBJ_FILES_DEBUG) | $(TARGET_DIR)
	$(CXX) $(DEBUG_FLAGS) -o $(TARGET_DIR)/$@ $^ $(LDFLAGS) 

lu-serial: $(OBJ_FILES_SERIAL) | $(TARGET_DIR)
	$(CXX) $(SERIAL_FLAGS) -o $(TARGET_DIR)/$@ $^ $(LDFLAGS) 

runp: lu
	$(TARGET_DIR)/$< $(MATRIX_SIZE) $(NUM_CORES)

runs: lu-serial
	$(TARGET_DIR)/$< $(MATRIX_SIZE) 1

check: lu
	$(CHECKER) $(TARGET_DIR)/$< $(MATRIX_SIZE) $(NUM_CORES)

view:
	$(VIEWER)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
$(OBJ_DIR)/debug/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)/debug
	$(CXX) $(DEBUG_FLAGS) -c -o $@ $<

$(OBJ_DIR)/serial/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)/serial
	$(CXX) $(SERIAL_FLAGS) -c -o $@ $<

$(TARGET_DIR) $(OBJ_DIR) $(OBJ_DIR)/serial $(OBJ_DIR)/debug:
	mkdir -p $@

clean:
	rm -r $(BUILD_DIR)