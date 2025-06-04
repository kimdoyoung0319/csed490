#include "core.h"

int main(int argc, char *argv[]) {
    initialize(argc, argv);
    execute();
    finalize();

    return 0;
}