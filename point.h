#pragma once

#include <math.h>
#include <vector>
#include <exception>
#include <iostream>

class point {
public:
  point(std::vector<double> &features) : dimension(features.size()), features(features) {}

  int get_cluster_id() const noexcept { return cluster_id; }

  void set_cluster_id(int cluster) {
    cluster_id = cluster;
  }

  int get_dimension() const noexcept { return dimension; }

  double get_feature(size_t index) const { return features[index]; }

  void debug() {
    std::cout << "Cluster Id: " << cluster_id << " => (";
    for (auto feat : features) {
      std::cout << feat << ", ";
    }
    std::cout << ")\n";
  }

private:
  int cluster_id{-1};
  std::vector<double> features;
  int dimension;
};