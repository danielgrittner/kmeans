#include "kmeans.h"

#include <vector>

int main() {
  // Initial test case:
  std::vector<std::vector<double>> dataset = {
          // A      B
          {1.0, 1.0},
          {1.5, 2.0},
          {3.0, 4.0},
          {5.0, 7.0},
          {3.5, 5.0},
          {4.5, 5.0},
          {3.5, 4.5}
  };
  int k = 2;
  int max_iterations = 2;

  kmeans kmeans(k, max_iterations, dataset);
  kmeans.run();

  return 0;
}
