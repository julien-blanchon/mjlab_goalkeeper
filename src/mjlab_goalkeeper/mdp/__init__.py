"""MDP components for goalkeeper environment."""

# Import custom goalkeeper components
from mjlab_goalkeeper.mdp.commands import (
    PenaltyKickCommand,
    PenaltyKickCommandCfg,
)
from mjlab_goalkeeper.mdp.events import (
    reset_robot_on_goal_line,
)
from mjlab_goalkeeper.mdp.observations import (
    target_interception_x_position,
)
from mjlab_goalkeeper.mdp.rewards import (
    base_height_reward,
    joint_limits_penalty,
    position_to_target_x,
    posture_reward,
    stay_on_goal_line_y,
    track_lateral_velocity_to_target,
)
from mjlab_goalkeeper.mdp.terminations import (
    base_too_low,
    fell_over,
)
from mjlab_goalkeeper.mdp.curriculums import (
    terrain_levels_vel,
)

__all__ = [
    # Custom commands
    "PenaltyKickCommand",
    "PenaltyKickCommandCfg",
    # Custom events
    "reset_robot_on_goal_line",
    # Custom observations
    "target_interception_x_position",
    # Custom curriculums
    "terrain_levels_vel",
    # Custom rewards
    "track_lateral_velocity_to_target",
    "position_to_target_x",
    "stay_on_goal_line_y",
    "base_height_reward",
    "posture_reward",
    "joint_limits_penalty",
    # Custom terminations
    "fell_over",
    "base_too_low",
]
