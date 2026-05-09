import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

parser = argparse.ArgumentParser(description="PPO Configuration")

parser.add_argument(
    "--config_name", type=str, default="rlvr_megatron_config", help="Name of the PPO configuration."
)
args = parser.parse_args()


def make_ppo_config():

    config_path = "."
    config_name = args.config_name

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline():

    ppo_config = make_ppo_config()

    init()

    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    test_ppo_pipeline()
