#pragma once
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include "weave_sim.hpp"

struct Solution {
    double score;
    std::vector<std::vector<int>> partition;
    std::vector<Job> job_deployment;
    std::vector<std::string> meta_iteration;
    
    Solution() : score(-std::numeric_limits<double>::infinity()) {}
};

class IntraGroupSolver {
private:
    std::vector<Job> original_jobs_;
    int sim_steps_;
    std::vector<std::string> idx_2_job_id_;
    
    // Partition generation
    std::vector<std::vector<std::vector<int>>> generatePartitions(int n);
    void generatePartitionsHelper(const std::vector<int>& elements, 
                                 std::vector<std::vector<std::vector<int>>>& result);
    
    // Composition generation
    std::vector<std::vector<int>> generateCompositions(int n, int k, int min_val);
    void generateCompositionsHelper(int n, int k, int min_val, 
                                   std::vector<int>& current, 
                                   std::vector<std::vector<int>>& result);
    
    // Permutation generation (distinct permutations)
    std::vector<std::vector<std::string>> generateDistinctPermutations(
        const std::vector<std::string>& elements);
    void generateDistinctPermutationsHelper(
        std::vector<std::string>& elements, int start, 
        std::vector<std::vector<std::string>>& result);
    
    // Meta iteration generation
    std::vector<std::vector<std::string>> generateMetaIterations(int max_meta_iter_len);
    
    // Job assembly
    std::vector<Job> assembleJobs(const std::vector<std::vector<int>>& partition);
    
public:
    IntraGroupSolver(const std::vector<Job>& jobs, int sim_steps = 1000);
    
    Solution solve(int max_meta_iter_len, 
                  const std::function<double(const std::vector<double>&, 
                                           const std::vector<double>&)>& score_func);
};
