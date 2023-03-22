// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <pva/pva.hpp>

#include <iostream>

int main(int, char *[]) {
  auto report = pva::openReport("./profile.pop");

  std::cout << "Example information from profile:\n"
            << "Number of compute sets: "
            << report.compilation().graph().numComputeSets() << "\n"
            << "Number of tiles on target: "
            << report.compilation().target().numTiles() << "\n"
            << "Version of Poplar used: " << report.poplarVersion().string()
            << "\n";

  return EXIT_SUCCESS;
}
