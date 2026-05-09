#include <chrono>
#include <iostream>
#include "brute_force_solver.hpp"
#include "weave_sim.hpp"

using namespace std::chrono;

std::ostream& operator<<(std::ostream& os, std::vector<std::string>& vec) {
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i != vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, std::vector<std::vector<int>>& vec) {
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << "[";
        for (int j = 0; j < vec[i].size(); j++) {
            std::cout << vec[i][j];
            if (j != vec[i].size() - 1)
                std::cout << ", ";
        }
        std::cout << "]";
        if (i != vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]";
    return os;
}

double score_func(std::vector<double> rollout_util, std::vector<double> train_util) {
    double rollout_waste_sum = 0, train_waste_sum = 0;
    for (int i = 0; i < rollout_util.size(); i++)
        rollout_waste_sum += (1 - rollout_util[i]);
    for (int i = 0; i < train_util.size(); i++)
        train_waste_sum += (1 - train_util[i]);
    return -1.0 * (rollout_waste_sum * 0.3 + train_waste_sum);
}

int main() {
    Job job_A("A", 5, 5, {}, {}), job_B("B", 5, 2.5, {}, {}),job_C("C", 5, 2.5, {}, {});
    std::vector<Job> jobs({job_A, job_B, job_C});
    IntraGroupSolver solver(jobs, 100);
    auto start = high_resolution_clock::now();
    auto ret = solver.solve(6, score_func);
    auto end = high_resolution_clock::now();
    auto duration = (double)duration_cast<microseconds>(end - start).count() / 1000000.0;
    std::cout << " meta_iteration: " << ret.meta_iteration\
        << " partition: " << ret.partition << " score: " << ret.score << std::endl;
    printf("Time to solve = %.3fs\n", duration);
    return 0;
}
