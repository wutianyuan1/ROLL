#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "weave_sim.hpp"
#include "brute_force_solver.hpp"

namespace py = pybind11;

// Wrapper for Solution struct
struct PySolution {
    double score;
    std::vector<std::vector<int>> partition;
    std::vector<Job> job_deployment;
    std::vector<std::string> meta_iteration;

    PySolution() = default;
    
    PySolution(const Solution& sol) 
        : score(sol.score), partition(sol.partition),
          job_deployment(sol.job_deployment), meta_iteration(sol.meta_iteration) {}
};

PYBIND11_MODULE(global_scheduler_cpp, m) {
    py::class_<Job>(m, "Job")
        .def(py::init<const std::string&, double, double, 
                      const std::vector<std::string>&, const std::vector<std::string>&>())
        .def_readwrite("job_id", &Job::job_id)
        .def_readwrite("t_rollout", &Job::t_rollout)
        .def_readwrite("t_train", &Job::t_train)
        .def_readwrite("rollout_nodes", &Job::rollout_nodes)
        .def_readwrite("train_nodes", &Job::train_nodes);
    
    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readwrite("rollout_busy_times", &SimulationResult::rollout_busy_times)
        .def_readwrite("train_busy_times", &SimulationResult::train_busy_times)
        .def_readwrite("rollout_utils", &SimulationResult::rollout_utils)
        .def_readwrite("train_utils", &SimulationResult::train_utils);
    
    py::class_<WeaveSimulator>(m, "WeaveSimulator")
        .def(py::init<const std::vector<Job>&, const std::vector<std::string>&>())
        .def("simulate_run", &WeaveSimulator::simulate_run);

    py::class_<PySolution>(m, "Solution")
        .def(py::init<>())
        .def_readwrite("score", &PySolution::score)
        .def_readwrite("partition", &PySolution::partition)
        .def_readwrite("job_deployment", &PySolution::job_deployment)
        .def_readwrite("meta_iteration", &PySolution::meta_iteration);
    
    py::class_<IntraGroupSolver>(m, "IntraGroupSolver")
        .def(py::init<const std::vector<Job>&, int>(), 
             py::arg("jobs"), py::arg("sim_steps") = 1000)
        .def("solve", [](IntraGroupSolver& solver, int max_meta_iter_len,
                        const std::function<double(const std::vector<double>&, 
                                                 const std::vector<double>&)>& score_func) {
            Solution result = solver.solve(max_meta_iter_len, score_func);
            return PySolution(result);
        });
}
