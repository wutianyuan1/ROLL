#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>

struct Job {
    std::string job_id;
    double t_rollout;
    double t_train;
    std::vector<std::string> rollout_nodes;
    std::vector<std::string> train_nodes;
    
    Job(const std::string& id, double rollout_time, double train_time, 
        const std::vector<std::string>& r_nodes, const std::vector<std::string>& t_nodes)
        : job_id(id), t_rollout(rollout_time), t_train(train_time), 
          rollout_nodes(r_nodes), train_nodes(t_nodes) {}
};

struct SimulationResult {
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<double, double>>>> rollout_busy_times;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::pair<double, double>>>> train_busy_times;
    std::vector<double> rollout_utils;
    std::vector<double> train_utils;
};

class WeaveSimulator {
private:
    std::vector<Job> jobs_;
    std::vector<std::string> meta_iter_cycle_;
    std::vector<std::string> all_rollout_nodes_;
    std::vector<std::string> all_train_nodes_;
    std::unordered_map<std::string, int> job_index_map_;
    
    double calculate_utilization(const std::vector<std::pair<double, double>>& busy_times) const;
    
public:
    WeaveSimulator(const std::vector<Job>& jobs, const std::vector<std::string>& meta_iter_cycle);
    
    SimulationResult simulate_run(int n_meta_iters);
};
