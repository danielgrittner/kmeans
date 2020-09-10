#pragma once

#include "point.h"

#include <vector>
#include <string>
#include <iostream> // FIXME: DEBUG

struct cluster {
  const int cluster_id;
  std::vector<double> features;

  // Calculates the euclidean norm between two points.
  double compute_distance_to_cluster(const point &point) {
    if (point.get_dimension() != features.size()) {
      throw std::runtime_error("Illegal arguments - compute_distance_between_points: Different dimensions provided.");
    }

    // Euclidean norm:
    double dist = 0;
    for (size_t i = 0; i < point.get_dimension(); i++) {
      double subtraction = point.get_feature(i) - features[i];
      dist += subtraction * subtraction;
    }

    return sqrt(dist);
  }

  // TODO: remove again
  void print_cluster() {
    std::cout << "Cluster id: " << cluster_id << "\n";
    std::cout << "Features: ";
    for (auto feature : features) {
      std::cout << feature << ", ";
    }
    std::cout << "\n";
  }
};

/**
 * Resources:
 * https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
 * https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
 * http://mnemstudio.org/clustering-k-means-example-1.htm
 */
class kmeans {
public:

  kmeans(int k, int max_iterations, std::vector<std::vector<double>> &dataset);

  // TODO: load dataset from file
  // kmeans(int k, int max_iterations, std::string dataset_file_path);

  /**
   * Performs the clustering of the dataset.
   */
  void run();

  void save_model();

  int determine_nearest_cluster(const point &point);

private:

  void debug(); // TODO: remove again

  void init_centroids(double min, double max, size_t dimension);

  void update_centroids();

  // Defines the number of clusters.
  const int k;
  const int max_num_of_iterations;
  const int dimension;

  std::vector<point> data_points;
  std::vector<cluster> clusters;
};
