// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <pva/pva.hpp>

int main(int, char *[]) {
  auto report = pva::preview::openReport("./profile.pop");

  std::cout << "Example information from profile:"
            << "\nNumber of compute sets: "
            << report.compilation().graph().numComputeSets()
            << "\nNumber of tiles on target: "
            << report.compilation().target().numTiles()
            << "\nVersion of Poplar used: " << report.poplarVersion().string()
            << "\n";

  return EXIT_SUCCESS;
}
