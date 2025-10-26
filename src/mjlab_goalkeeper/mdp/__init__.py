"""MDP components for goalkeeper environment."""

# Import custom goalkeeper components
from mjlab_goalkeeper.mdp.commands import (
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
    goal_scored_detection,
    hand_position,
    robot_ball_contact,
)
from mjlab_goalkeeper.mdp.rewards import (
    goal_prevented_by_robot,
    goal_prevented_reward,
    hand_to_ball_distance,
    is_alive,
    post_contact_stabilization,
    robot_ball_contact_reward,
    feet_air_time,
    foot_clearance_reward,
    feet_slide,
    track_ang_vel_exp,
    track_lin_vel_exp,
)
from mjlab_goalkeeper.mdp.terminations import (
    goal_scored_termination,
)
from mjlab_goalkeeper.mdp.curriculums import (
    terrain_levels_vel,
)

__all__ = [
    # Custom commands
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
    "feet_air_time",
    "foot_clearance_reward",
    "feet_slide",
    "track_ang_vel_exp",
    "track_lin_vel_exp",
    # Custom curriculums
    "terrain_levels_vel",
]
