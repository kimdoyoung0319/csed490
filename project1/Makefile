# TODO: Add runp, runs, check, view rules.

.PHONY: all clean

BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
TARGET_DIR := $(BUILD_DIR)/target
SRC_DIR := src

CXX := g++
CXXFLAGS := -Wall -O2 -fopenmp
LDFLAGS := -lrt
DEBUG_FLAGS := -Wall -O0 -g -fopenmp
SERIAL_FLAGS := -Wall -O2 -g

TARGETS := lu lu-serial lu-debug

SRC_FILES := $(SRC_DIR)/lu.cpp $(SRC_DIR)/decomp.cpp $(SRC_DIR)/gen.cpp $(SRC_DIR)/matrix.cpp
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
OBJ_FILES_DEBUG := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/debug/%.o, $(SRC_FILES))
OBJ_FILES_SERIAL := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/serial/%.o, $(SRC_FILES))

all: $(TARGETS)

lu: $(OBJ_FILES) | $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET_DIR)/$@ $^ $(LDFLAGS) 

lu-debug: $(OBJ_FILES_SERIAL) | $(TARGET_DIR)
	$(CXX) $(DEBUG_FLAGS) -o $(TARGET_DIR)/$@ $^ $(LDFLAGS) 

lu-serial: $(OBJ_FILES_SERIAL) | $(TARGET_DIR)
	$(CXX) $(SERIAL_FLAGS) -o $(TARGET_DIR)/$@ $^ $(LDFLAGS) 

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