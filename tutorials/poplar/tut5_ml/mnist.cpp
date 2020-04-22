// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include "mnist.h"
#include "assert.h"
#include <fstream>
#include <iostream>

// IDX file format
// Simple format for vectors and multidimensional matrices of various
// numeric types. The basic format is:

// TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
// ........
// xxxx     unsigned byte   ??               label
// The labels values are 0 to 9.
//
// TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  60000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel
// Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background
// (white), 255 means foreground (black).

static unsigned int readBigEndianUnsignedInt(std::ifstream &file) {
  unsigned int i = 0;
  // read 4  bytes from wherever the get pointer of file is located
  file.read((char *)&i, sizeof(i));
  // this will be in big endian format ( A B C D) / natural order
  // convert this into a little endian format by reordering the bytes
  // to ( D C B A)
  return ((i & 0x000000ff) << 24) | ((i & 0x0000ff00) << 8) |
         ((i & 0x00ff0000) >> 8) | ((i & 0xff000000) >> 24);
}

// All integers in the database are stored in BIG endian format
// All Intel and other low-endian machines must flip the bytes of the header
std::vector<unsigned> readMNISTLabels(const char *fname) {
  std::ifstream file(fname, std::ios::binary);
  if (!file.is_open()) {
    return std::vector<unsigned>();
  }

  // read first two entries 4 bytes at a time and convert
  // from big-endian to little endian
  unsigned magic_number = readBigEndianUnsignedInt(file);
  assert(magic_number == 2049);
  unsigned numberOfImages = readBigEndianUnsignedInt(file);
  std::vector<unsigned> arr(numberOfImages);
  // read labels byte by byte
  for (int i = 0; i < numberOfImages; ++i) {
    // temp is of type char (1 byte)
    unsigned char temp = 0;
    file.read((char *)&temp, sizeof(temp));
    arr[i] = temp;
  }
  return arr;
}

std::vector<float> readMNISTData(unsigned &numberOfImages, unsigned &imageSize,
                                 const char *fname) {
  std::ifstream file(fname, std::ios::binary);
  if (!file.is_open()) {
    return std::vector<float>();
  }

  unsigned magic_number = readBigEndianUnsignedInt(file);
  assert(magic_number == 2051);
  numberOfImages = readBigEndianUnsignedInt(file);
  unsigned n_rows = readBigEndianUnsignedInt(file);
  unsigned n_cols = readBigEndianUnsignedInt(file);
  imageSize = n_rows * n_cols;

  std::vector<float> pixels;
  pixels.reserve(numberOfImages * imageSize);

  std::vector<unsigned char> buf(imageSize);
  for (int i = 0; i < numberOfImages; ++i) {
    file.read((char *)buf.data(), imageSize); // read 784 bytes in a go
    for (auto &pixel : buf) {
      // re-cast byte stream from char to float and normalize
      // to set pixel values in [0,1]
      pixels.push_back(static_cast<float>(pixel) / 256);
    }
  }
  return pixels;
}
