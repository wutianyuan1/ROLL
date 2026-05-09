#include "brute_force_solver.hpp"
#include <iostream>
#include <unordered_map>
#include <set>

IntraGroupSolver::IntraGroupSolver(const std::vector<Job>& jobs, int sim_steps)
    : original_jobs_(jobs), sim_steps_(sim_steps) {
    
    // Build index to job_id mapping
    for (const auto& job : original_jobs_) {
        idx_2_job_id_.push_back(job.job_id);
    }
}

// Partition generation using iterative approach for better performance
std::vector<std::vector<std::vector<int>>> IntraGroupSolver::generatePartitions(int n) {
    std::vector<std::vector<std::vector<int>>> all_partitions;
    
    if (n == 0) {
        all_partitions.push_back({});
        return all_partitions;
    }
    
    // Initialize with first element
    std::vector<std::vector<std::vector<int>>> partitions = {{{0}}};
    
    // Add elements one by one
    for (int i = 1; i < n; ++i) {
        std::vector<std::vector<std::vector<int>>> new_partitions;
        
        for (const auto& partition : partitions) {
            // Option 1: put element i in a new subset
            auto new_partition1 = partition;
            new_partition1.push_back({i});
            new_partitions.push_back(new_partition1);
            
            // Option 2: put element i in each existing subset
            for (size_t j = 0; j < partition.size(); ++j) {
                auto new_partition2 = partition;
                new_partition2[j].push_back(i);
                new_partitions.push_back(new_partition2);
            }
        }
        
        partitions = std::move(new_partitions);
    }
    
    // Normalize partitions (sort subsets and sort list of subsets)
    std::set<std::vector<std::vector<int>>> normalized_set;
    
    for (auto& partition : partitions) {
        // Sort each subset
        for (auto& subset : partition) {
            std::sort(subset.begin(), subset.end());
        }
        // Sort list of subsets lexicographically
        std::sort(partition.begin(), partition.end());
        normalized_set.insert(partition);
    }
    
    all_partitions.assign(normalized_set.begin(), normalized_set.end());
    return all_partitions;
}

// Composition generation using iterative approach
std::vector<std::vector<int>> IntraGroupSolver::generateCompositions(int n, int k, int min_val) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    generateCompositionsHelper(n, k, min_val, current, result);
    return result;
}

void IntraGroupSolver::generateCompositionsHelper(int n, int k, int min_val, 
                                                std::vector<int>& current, 
                                                std::vector<std::vector<int>>& result) {
    if (k == 1) {
        if (n >= min_val) {
            current.push_back(n);
            result.push_back(current);
            current.pop_back();
        }
        return;
    }
    
    for (int i = min_val; i <= n - min_val * (k - 1); ++i) {
        current.push_back(i);
        generateCompositionsHelper(n - i, k - 1, min_val, current, result);
        current.pop_back();
    }
}

// Distinct permutations generation
std::vector<std::vector<std::string>> IntraGroupSolver::generateDistinctPermutations(
    const std::vector<std::string>& elements) {
    
    std::vector<std::vector<std::string>> result;
    if (elements.empty()) return result;
    
    std::vector<std::string> mutable_elements = elements;
    generateDistinctPermutationsHelper(mutable_elements, 0, result);
    return result;
}

void IntraGroupSolver::generateDistinctPermutationsHelper(
    std::vector<std::string>& elements, int start, 
    std::vector<std::vector<std::string>>& result) {
    
    if (start == elements.size() - 1) {
        result.push_back(elements);
        return;
    }
    
    std::unordered_set<std::string> seen;
    for (int i = start; i < elements.size(); ++i) {
        if (seen.find(elements[i]) != seen.end()) continue;
        seen.insert(elements[i]);
        
        std::swap(elements[start], elements[i]);
        generateDistinctPermutationsHelper(elements, start + 1, result);
        std::swap(elements[start], elements[i]);
    }
}

// Meta iteration generation
std::vector<std::vector<std::string>> IntraGroupSolver::generateMetaIterations(int max_meta_iter_len) {
    int n_jobs = original_jobs_.size();
    std::vector<std::vector<std::string>> all_meta_iters;
    
    for (int meta_iter_len = n_jobs; meta_iter_len <= max_meta_iter_len; ++meta_iter_len) {
        auto compositions = generateCompositions(meta_iter_len, n_jobs, 1);
        
        for (const auto& composition : compositions) {
            std::vector<std::string> job_comb;
            for (int i = 0; i < n_jobs; ++i) {
                for (int j = 0; j < composition[i]; ++j) {
                    job_comb.push_back(idx_2_job_id_[i]);
                }
            }
            
            auto permutations = generateDistinctPermutations(job_comb);
            all_meta_iters.insert(all_meta_iters.end(), permutations.begin(), permutations.end());
        }
    }
    
    return all_meta_iters;
}

// Job assembly
std::vector<Job> IntraGroupSolver::assembleJobs(const std::vector<std::vector<int>>& partition) {
    std::vector<Job> ret_jobs = original_jobs_;
    
    for (size_t rollout_node_id = 0; rollout_node_id < partition.size(); ++rollout_node_id) {
        for (int job_idx : partition[rollout_node_id]) {
            ret_jobs[job_idx].rollout_nodes = {"RN-" + std::to_string(rollout_node_id)};
            ret_jobs[job_idx].train_nodes = {"TN"};
        }
    }
    
    return ret_jobs;
}

// Main solver
Solution IntraGroupSolver::solve(int max_meta_iter_len, 
                               const std::function<double(const std::vector<double>&, 
                                                        const std::vector<double>&)>& score_func) {
    
    if (max_meta_iter_len < original_jobs_.size()) {
        throw std::invalid_argument("max_meta_iter_len must be >= number of jobs");
    }
    
    auto all_partitions = generatePartitions(original_jobs_.size());
    auto meta_iterations = generateMetaIterations(max_meta_iter_len);
    
    std::cout << "Generated " << all_partitions.size() << " partitions and " 
              << meta_iterations.size() << " meta iterations" << std::endl;
    
    Solution best_solution;
    
    int counter = 0;
    for (const auto& partition : all_partitions) {
        auto job_deployment = assembleJobs(partition);
        
        for (const auto& meta_iter : meta_iterations) {
            WeaveSimulator sim(job_deployment, meta_iter);
            auto result = sim.simulate_run(sim_steps_);
            
            double score = score_func(result.rollout_utils, result.train_utils);
            
            if (++counter % 1000 == 0) {
                std::cout << "Processed " << counter << " combinations, current best score: " 
                          << best_solution.score << std::endl;
            }
            
            if (score > best_solution.score) {
                best_solution.score = score;
                best_solution.partition = partition;
                best_solution.job_deployment = job_deployment;
                best_solution.meta_iteration = meta_iter;
            }
        }
    }
    
    std::cout << "Best score: " << best_solution.score << std::endl;
    return best_solution;
}