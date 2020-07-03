// Copyright 2020 Graphcore Ltd.
#include <iostream>
#include "compile_time_version.h"

int main(int argc, char** argv) {
    std::cout << getPluginVersion() << "\n";
    return 0;
}