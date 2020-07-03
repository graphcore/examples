// Copyright 2020 Graphcore Ltd.
#ifndef COMPILE_TIME_VERSION_H
#define COMPILE_TIME_VERSION_H

#include <poplar/Graph.hpp>

inline std::string getPluginVersion() {
    // Replace parenthesis and space in version string so
    // we can easily use the results as a variable in a
    // Makefile and on the compiler command line:
    std::string version = poplar::versionString();
    for (char &c : version) {
        if (c == '(' || c == ')' || c == ' ') {
            c = '-';
        }
    }
    return version;
}

#ifdef STATIC_VERSION
static void __attribute__ ((constructor)) shared_object_init() {
  const std::string runtimeVersion = getPluginVersion();
  if (runtimeVersion != STATIC_VERSION) {
    std::cerr << "ERROR: plug-in version mismatch\n"
              << "STATIC VERSION: " << STATIC_VERSION << " RUN-TIME VERSION: " << runtimeVersion << "\n"
              << "Please recompile the custom operators with the sdk that you are using by runing `make' \n";
    exit(1);
  }
}
#endif

#endif
