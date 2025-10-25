"""MDP components for goalkeeper environment."""

# Import all base MDP components from mjlab
from mjlab.envs.mdp import *  # noqa: F403

# Import velocity-specific components
from mjlab.tasks.velocity.mdp import *  # noqa: F403

# Import custom goalkeeper components
from mjlab_goalkeeper.mdp.commands import (
    FootballThrowCommand,
    FootballThrowCommandCfg,
    PenaltyKickCommand,
    PenaltyKickCommandCfg,
)
from mjlab_goalkeeper.mdp.events import (
    reset_ball_on_circle,
    reset_robot_on_goal_line,
    update_ball_circular_motion,
)
from mjlab_goalkeeper.mdp.observations import (
    entity_contact_detected,
    entity_contact_force,
    entity_position,
    entity_relative_position,
    entity_velocity,
    goal_prevented_reward,
    goal_scored_detection,
    goal_scored_termination,
    hand_position,
    robot_ball_contact,
    robot_ball_contact_reward,
)
from mjlab_goalkeeper.mdp.rewards import (
    goal_prevented_by_robot,
    hand_to_ball_distance,
    is_alive,
    post_contact_stabilization,
)

__all__ = [
    # Custom commands
    "FootballThrowCommand",
    "FootballThrowCommandCfg",
    "PenaltyKickCommand",
    "PenaltyKickCommandCfg",
    # Custom events
    "update_ball_circular_motion",
    "reset_ball_on_circle",
    "reset_robot_on_goal_line",
    # Custom observations
    "entity_position",
    "entity_velocity",
    "entity_relative_position",
    "entity_contact_force",
    "entity_contact_detected",
    "goal_scored_detection",
    "robot_ball_contact",
    "hand_position",
    # Reward/termination wrappers
    "robot_ball_contact_reward",
    "goal_prevented_reward",
    "goal_scored_termination",
    # Custom rewards
    "hand_to_ball_distance",
    "post_contact_stabilization",
    "is_alive",
    "goal_prevented_by_robot",
]
