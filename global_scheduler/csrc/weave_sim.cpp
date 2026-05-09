#include "weave_sim.hpp"
#include <iostream>
#include <limits>
#include <unordered_set>

WeaveSimulator::WeaveSimulator(const std::vector<Job>& jobs, 
                               const std::vector<std::string>& meta_iter_cycle)
    : jobs_(jobs), meta_iter_cycle_(meta_iter_cycle) {
    
    // Build job index map for fast lookup
    for (int i = 0; i < jobs_.size(); ++i) {
        job_index_map_[jobs_[i].job_id] = i;
    }
    
    // Collect all unique nodes
    std::unordered_set<std::string> rollout_nodes_set, train_nodes_set;
    for (const auto& job : jobs_) {
        for (const auto& node : job.rollout_nodes) {
            rollout_nodes_set.insert(node);
        }
        for (const auto& node : job.train_nodes) {
            train_nodes_set.insert(node);
        }
    }
    
    all_rollout_nodes_.assign(rollout_nodes_set.begin(), rollout_nodes_set.end());
    all_train_nodes_.assign(train_nodes_set.begin(), train_nodes_set.end());
}

double WeaveSimulator::calculate_utilization(const std::vector<std::pair<double, double>>& busy_times) const {
    if (busy_times.empty()) return 0.0;
    
    double min_t = std::numeric_limits<double>::max();
    double max_t = std::numeric_limits<double>::lowest();
    double busy_total = 0.0;
    
    for (const auto& interval : busy_times) {
        min_t = std::min(min_t, interval.first);
        max_t = std::max(max_t, interval.second);
        busy_total += (interval.second - interval.first);
    }
    
    if (max_t - min_t < 1e-9) return 0.0;
    return busy_total / (max_t - min_t);
}

SimulationResult WeaveSimulator::simulate_run(int n_meta_iters) {
    SimulationResult result;
    
    // Initialize busy times data structures
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<double, double>>>> rollout_busy_times;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<double, double>>>> train_busy_times;
    
    for (const auto& node : all_rollout_nodes_) {
        rollout_busy_times[node] = {};
    }
    for (const auto& node : all_train_nodes_) {
        train_busy_times[node] = {};
    }
    
    int cycle_len = meta_iter_cycle_.size();
    
    for (int meta_iter = 0; meta_iter < n_meta_iters; ++meta_iter) {
        for (int i = 0; i < cycle_len; ++i) {
            const std::string& job_id = meta_iter_cycle_[i];
            int job_idx = job_index_map_.at(job_id);
            const Job& job = jobs_[job_idx];
            
            // Schedule rollout phase
            double prev_rollout_end = 0.0;
            for (int j = 1; j <= cycle_len; ++j) {
                int cur_idx = (i + cycle_len - j) % cycle_len;
                const std::string& cur_job_id = meta_iter_cycle_[cur_idx];
                int cur_job_idx = job_index_map_.at(cur_job_id);
                const Job& cur_job = jobs_[cur_job_idx];
                
                // Check for shared rollout nodes
                bool shares_rollout_node = false;
                for (const auto& node : cur_job.rollout_nodes) {
                    if (std::find(job.rollout_nodes.begin(), job.rollout_nodes.end(), node) != job.rollout_nodes.end()) {
                        shares_rollout_node = true;
                        break;
                    }
                }
                
                if (shares_rollout_node && !cur_job.rollout_nodes.empty()) {
                    const auto& cur_busy_times = rollout_busy_times[cur_job.rollout_nodes[0]][cur_job_id];
                    double cur_end = cur_busy_times.empty() ? 0.0 : cur_busy_times.back().second;
                    prev_rollout_end = std::max(prev_rollout_end, cur_end);
                }
            }
            
            // Get last train end time for this job
            double last_train_end = 0.0;
            if (!job.train_nodes.empty()) {
                const auto& train_times = train_busy_times[job.train_nodes[0]][job_id];
                last_train_end = train_times.empty() ? 0.0 : train_times.back().second;
            }
            
            double t_rollout_begin = std::max(last_train_end, prev_rollout_end);
            
            // Add rollout intervals
            for (const auto& node : job.rollout_nodes) {
                rollout_busy_times[node][job_id].emplace_back(
                    t_rollout_begin, t_rollout_begin + job.t_rollout
                );
            }
            
            // Schedule train phase
            double prev_train_end = 0.0;
            for (int j = 1; j <= cycle_len; ++j) {
                int cur_idx = (i + cycle_len - j) % cycle_len;
                const std::string& cur_job_id = meta_iter_cycle_[cur_idx];
                int cur_job_idx = job_index_map_.at(cur_job_id);
                const Job& cur_job = jobs_[cur_job_idx];
                
                // Check for shared train nodes
                bool shares_train_node = false;
                for (const auto& node : cur_job.train_nodes) {
                    if (std::find(job.train_nodes.begin(), job.train_nodes.end(), node) != job.train_nodes.end()) {
                        shares_train_node = true;
                        break;
                    }
                }
                
                if (shares_train_node && !cur_job.train_nodes.empty()) {
                    const auto& cur_busy_times = train_busy_times[cur_job.train_nodes[0]][cur_job_id];
                    double cur_end = cur_busy_times.empty() ? 0.0 : cur_busy_times.back().second;
                    prev_train_end = std::max(prev_train_end, cur_end);
                }
            }
            
            // Get last rollout end time for this job
            double last_rollout_end = 0.0;
            if (!job.rollout_nodes.empty()) {
                const auto& rollout_times = rollout_busy_times[job.rollout_nodes[0]][job_id];
                last_rollout_end = rollout_times.empty() ? 0.0 : rollout_times.back().second;
            }
            
            double t_train_begin = std::max(last_rollout_end, prev_train_end);
            
            // Add train intervals
            for (const auto& node : job.train_nodes) {
                train_busy_times[node][job_id].emplace_back(
                    t_train_begin, t_train_begin + job.t_train
                );
            }
        }
    }
    
    // Calculate utilizations
    for (const auto& node : all_rollout_nodes_) {
        std::vector<std::pair<double, double>> all_intervals;
        for (const auto& job_entry : rollout_busy_times[node]) {
            all_intervals.insert(all_intervals.end(), 
                               job_entry.second.begin(), job_entry.second.end());
        }
        result.rollout_utils.push_back(calculate_utilization(all_intervals));
    }
    
    for (const auto& node : all_train_nodes_) {
        std::vector<std::pair<double, double>> all_intervals;
        for (const auto& job_entry : train_busy_times[node]) {
            all_intervals.insert(all_intervals.end(), 
                               job_entry.second.begin(), job_entry.second.end());
        }
        result.train_utils.push_back(calculate_utilization(all_intervals));
    }
    
    result.rollout_busy_times = rollout_busy_times;
    result.train_busy_times = train_busy_times;
    return result;
}
