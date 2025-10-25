"""Velocity tracking environments for legged robots."""

import gymnasium as gym

gym.register(
    id="Mjlab-Goalkeeper",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.env_cfg:UnitreeG1FlatEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.config.rl_cfg:UnitreeG1PPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Goalkeeper-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.env_cfg:UnitreeG1FlatEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.config.rl_cfg:UnitreeG1PPORunnerCfg",
    },
)
