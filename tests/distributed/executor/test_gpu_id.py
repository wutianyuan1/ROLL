def test_gpu_id():
    ray.init(log_to_driver=True)

    resource_manager = ResourceManager()

    test_worker_config = WorkerConfig(name="test_worker", world_size=8)
    test_cluster: Any = Cluster(
        name=test_worker_config.name,
        resource_manager=resource_manager,
        worker_cls=TestDPWorker,
        worker_config=test_worker_config,
    )
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    res = test_cluster.add(x=x)
    print(res)
    assert res == [
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 4],
        [1, 0, 1, 5],
        [1, 0, 1, 6],
        [1, 0, 1, 7],
        [1, 0, 1, 8],
    ]


if __name__ == "__main__":
    test_gpu_id()
