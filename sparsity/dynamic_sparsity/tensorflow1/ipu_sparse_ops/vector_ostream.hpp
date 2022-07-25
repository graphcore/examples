// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <iostream>
namespace std {
  template <typename T>
  ostream& operator<< (ostream& out, const vector<T>& v) {
    if (!v.empty()) {
      out << " ";
      copy (v.begin(), v.end(), ostream_iterator<T>(out, " "));
    }
    return out;
  }

  template <typename T, typename U>
  ostream& operator<< (ostream& out, const std::pair<T,U>& p) {
    out << "(" << p.first << "," << p.second << ")";
  }
}

