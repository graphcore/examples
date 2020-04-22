// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef _MNIST_H_
#define _MNIST_H_
#include <vector>

std::vector<unsigned> readMNISTLabels(const char *fname);

std::vector<float> readMNISTData(unsigned &numberOfImages, unsigned &imageSize,
                                 const char *fname);

#endif //_MNIST_H_
