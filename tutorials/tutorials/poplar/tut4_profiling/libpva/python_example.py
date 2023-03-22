# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pva

report = pva.openReport("./profile.pop")

print("Example information from profile:")
print("Number of compute sets:", report.compilation.graph.numComputeSets)
print("Number of tiles on target:", report.compilation.target.numTiles)
print("Version of Poplar used:", report.poplarVersion.string)
