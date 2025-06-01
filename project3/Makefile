.PHONY: all run clean

SRC_DIR := src
BUILD_DIR := build
TARGET_DIR := $(BUILD_DIR)/target
OBJ_DIR := $(BUILD_DIR)/obj
INPUT_DIR := sample

CC := mpicc
RT := mpirun
LDFLAGS := -lm

FILES := main.c board.c
SRC_FILES := $(FILES:%=$(SRC_DIR)/%)
OBJ_FILES := $(patsubst %.c, $(OBJ_DIR)/%.o, $(FILES))

VALGRIND_SUPP := /usr/local/openmpi-5.0.7/share/openmpi/openmpi-valgrind.supp

all: project3

project3: $(OBJ_FILES) | $(TARGET_DIR)
	$(CC) \-o $(TARGET_DIR)/$@ $^ $(LDFLAGS)

run: project3
	$(RT) -np $(NP) $(TARGET_DIR)/$< < $(INPUT_DIR)/$(INPUT)

run-valgrind: project3
	$(RT) -np $(NP) valgrind --suppressions=$(VALGRIND_SUPP) $(TARGET_DIR)/$< < $(INPUT_DIR)/$(INPUT)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) \-c \-o $@ $<

$(TARGET_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -r $(BUILD_DIR)