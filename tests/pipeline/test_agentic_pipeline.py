import argparse
import json
import os
from dataclasses import asdict

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig

parser = argparse.ArgumentParser(description="PPO Configuration")

parser.add_argument(
    "--config_name", type=str, default="agentic_pipeline_config", help="Name of the PPO configuration."
)
args = parser.parse_args()


def make_ppo_config():
    config_path = "."
    config_name = args.config_name

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    print(cfg)
    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline():
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
    ppo_config = make_ppo_config()

    init()

    pipeline = AgenticPipeline(pipeline_config=ppo_config)

    pipeline.run()

    output_file = "ppo_pipeline.json"
    with open(output_file, "w") as f:
        json.dump(asdict(pipeline.state), f, ensure_ascii=False)


if __name__ == "__main__":
    test_ppo_pipeline()
