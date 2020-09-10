#include "kmeans.h"

#include <limits>
#include <algorithm>
#include <random>
#include <iterator>

void kmeans::init_centroids(double min, double max, size_t dimension) {
  // https://stackoverflow.com/questions/21102105/random-double-c11
  std::uniform_real_distribution<double> unif(min, max);
  std::random_device rand_device;
  std::mt19937 rand_engine(rand_device());

  for (int cluster_id = 0; cluster_id < k; cluster_id++) {
    // Random initial features.
    std::vector<double> init_features(dimension);
    std::generate(init_features.begin(), init_features.end(), [&rand_engine, &unif]() {
      return unif(rand_engine);
    });

    clusters.push_back({cluster_id, std::move(init_features)});
  }
}

kmeans::kmeans(const int k, const int max_iterations, std::vector<std::vector<double>> &dataset)
  : k(k), max_num_of_iterations(max_iterations), dimension(dataset.front().size()) {
  data_points.reserve(dataset.size());

  double max_value = std::numeric_limits<double>::min();
  double min_value = std::numeric_limits<double>::max();

  // Create the data points.
  for (auto &point_features : dataset) {
    if (dimension != point_features.size()) {
      throw std::runtime_error("Illegal dataset - points have different dimensions.");
    }
    data_points.emplace_back(point_features);

    // TODO: this can probably be optimized (or a different strategy could be applied
    //  for generating the initial centroid features)
    auto max_iter = std::max_element(point_features.begin(), point_features.end());
    auto min_iter = std::min_element(point_features.begin(), point_features.end());

    max_value = max_iter != point_features.end() && max_value < *max_iter ? *max_iter : max_value;
    min_value = min_iter != point_features.end() && min_value > *min_iter ? *min_iter : min_value;
  }

  init_centroids(min_value, max_value, dimension);
}

void kmeans::debug() {
  std::cout << "Centroids: \n";
  for (auto cluster : clusters) {
    cluster.print_cluster();
  }

  std::cout << "\n\nData points: \n";
  for (auto point : data_points) {
    point.debug();
  }
}

int kmeans::determine_nearest_cluster(const point &point) {
  int index_min_cluster = -1;
  double current_min_distance = std::numeric_limits<double>::max();

  for (auto &cluster : clusters) {
    auto dist = cluster.compute_distance_to_cluster(point);
    if (current_min_distance > dist) {
      current_min_distance = dist;
      index_min_cluster = cluster.cluster_id;
    }
  }

  return index_min_cluster;
}

void kmeans::update_centroids() {
  std::vector<int> num_of_points_per_cluster(clusters.size(), 0);

  // Clear the features of the clusters.
  for (auto &cluster : clusters) {
    std::fill(cluster.features.begin(), cluster.features.end(), 0);
  }

  // Now we accumulate the features from the assigned points.
  // TODO: This can be parallelized with tbb::parallel_for
  for (auto &point : data_points) {
    num_of_points_per_cluster[point.get_cluster_id()]++;
    auto &assigned_cluster = clusters[point.get_cluster_id()];
    for (size_t i = 0; i < dimension; i++) {
      assigned_cluster.features[i] += point.get_feature(i);
    }
  }

  // Then, we divide the accumulated features from each cluster and divide it by the number of assigned points.
  for (auto &cluster : clusters) {
    auto &assigned_points = num_of_points_per_cluster[cluster.cluster_id];
    for (size_t i = 0; i < dimension; i++) {
      cluster.features[i] /= assigned_points;
    }
  }
}

void kmeans::run() {
  // TODO:
  debug();

  int iters = max_num_of_iterations;
  while (iters--) {
    // Iterate through the points and assign a cluster
    for (auto &point : data_points) {
      int cluster_id = determine_nearest_cluster(point);
      point.set_cluster_id(cluster_id);
    }
    debug();

    // Update the centroids.
    update_centroids();
    debug(); // FIXME:
  }
}