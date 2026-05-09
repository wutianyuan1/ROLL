import itertools

import ray

from roll.distributed.executor.cluster import Cluster


class EnvironmentManager(Cluster):

    def initialize(self, pipeline_config, infer_cluster: Cluster, input_queue, output_queue, mode: str = "train"):
        infer_worker_iter = itertools.cycle(infer_cluster.workers)
        refs = []
        for worker in self.workers:
            infer_worker = next(infer_worker_iter)
            refs.append(
                worker.initialize.remote(
                    pipeline_config=pipeline_config,
                    infer_worker=infer_worker,
                    input_queue=input_queue,
                    output_queue=output_queue,
                    mode=mode,
                )
            )
        ray.get(refs)
