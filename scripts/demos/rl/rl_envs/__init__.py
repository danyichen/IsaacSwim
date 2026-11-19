"""Reinforcement learning environments and RSL-RL configuration."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacSwim-BlueROV-StationKeep-v0",
    entry_point=f"{__name__}.station_keep:StationKeepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.station_keep:StationKeepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StationKeepPPORunnerCfg",
    },
)


for _name, _cfg in [
    ("StationKeepTask", "StationTaskEnvCfg"),
    ("PipelineInspect", "PipelineEnvCfg"),
    ("TargetApproach", "TargetEnvCfg"),
    ("Dock", "DockEnvCfg"),
    ("SampleCollect", "SampleEnvCfg"),
    ("ValveTurn", "ValveEnvCfg"),
]:
    gym.register(
        id=f"IsaacSwim-BlueROV-{_name}-v0",
        entry_point=f"{__name__}.tasks:TaskEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StationKeepPPORunnerCfg",
        },
    )


def runner_cfg(agent_cfg) -> dict:
    try:
        from importlib.metadata import version

        from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    except ImportError:
        return agent_cfg.to_dict()
    return handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib")).to_dict()
