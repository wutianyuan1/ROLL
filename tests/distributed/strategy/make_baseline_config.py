from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.pipeline.rlvr.rlvr_config import RLVRConfig


def make_baseline_config(config_path, config_name):

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config
