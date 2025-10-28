"""Custom termination functions for goalkeeper environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fell_over(
    env: ManagerBasedRlEnv,
    limit_angle: float = math.radians(60.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if robot falls over (tilted beyond limit angle).

    Uses projected gravity to detect orientation. When upright, projected
    gravity is (0, 0, -1). When tilted, the Z component deviates from -1.

    Args:
        env: Environment.
        limit_angle: Maximum tilt angle in radians before termination (default: 60°).
        asset_cfg: Robot configuration.

    Returns:
        Bool tensor of shape (num_envs,). True if fallen, False otherwise.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Projected gravity in body frame
    proj_gravity = asset.data.projected_gravity_b  # (num_envs, 3)

    # When upright: proj_gravity[2] ≈ -1
    # When tilted by angle θ: proj_gravity[2] ≈ -cos(θ)
    # Termination threshold: cos(limit_angle)
    threshold = math.cos(limit_angle)

    # Terminate if Z component is above threshold (less negative = more tilted)
    fallen = proj_gravity[:, 2] > -threshold

    return fallen


def base_too_low(
    env: ManagerBasedRlEnv,
    min_height: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if robot base drops too low (crouched/collapsed).

    This prevents the robot from crawling or collapsing on the ground.

    Args:
        env: Environment.
        min_height: Minimum allowed base height in meters (default: 0.4m).
        asset_cfg: Robot configuration.

    Returns:
        Bool tensor of shape (num_envs,). True if too low, False otherwise.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get base Z position (height above ground)
    base_pos = asset.data.root_link_pos_w  # (num_envs, 3)
    env_origins = env.scene.env_origins  # (num_envs, 3)

    # Height relative to environment origin
    height = base_pos[:, 2] - env_origins[:, 2]  # (num_envs,)

    # Terminate if height below threshold
    too_low = height < min_height

    return too_low


def goal_scored(
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    """Terminate episode if ball enters the goal.

    This can be used as a termination condition (episode failure) or
    just tracked for metrics. By default, we'll track it but not terminate
    to allow the robot to learn from failed attempts.

    Args:
        env: Environment.
        ball_cfg: Ball configuration.
        goal_cfg: Goal configuration.

    Returns:
        Bool tensor of shape (num_envs,). True if goal scored, False otherwise.
    """
    ball: Entity = env.scene[ball_cfg.name]
    goal: Entity = env.scene[goal_cfg.name]

    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)
    goal_pos = goal.data.root_link_pos_w  # (num_envs, 3)

    # Goal dimensions (from goal entity)
    goal_width = 3.5
    goal_height = 2.0
    goal_line_y = -0.5
    goal_depth = 1.0

    # Check if ball is in goal
    x_in_goal = torch.abs(ball_pos[:, 0] - goal_pos[:, 0]) < (goal_width / 2)
    y_behind_line = ball_pos[:, 1] < goal_line_y
    y_in_goal = ball_pos[:, 1] > (goal_line_y - goal_depth)
    z_in_goal = (ball_pos[:, 2] > 0.0) & (ball_pos[:, 2] < goal_height)

    goal_scored_flag = x_in_goal & y_behind_line & y_in_goal & z_in_goal

    return goal_scored_flag


def robot_out_of_bounds(
    env: ManagerBasedRlEnv,
    x_limit: float = 2.5,
    y_limit_forward: float = 5.0,
    y_limit_backward: float = -2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if robot moves too far from goal area.

    This prevents the robot from wandering away or chasing the ball too far.

    Args:
        env: Environment.
        x_limit: Maximum lateral distance from center (meters).
        y_limit_forward: Maximum distance forward from goal (meters).
        y_limit_backward: Maximum distance backward from goal (meters, negative).
        asset_cfg: Robot configuration.

    Returns:
        Bool tensor of shape (num_envs,). True if out of bounds, False otherwise.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get robot position relative to environment origin
    robot_pos = asset.data.root_link_pos_w  # (num_envs, 3)
    env_origins = env.scene.env_origins  # (num_envs, 3)

    local_pos = robot_pos - env_origins  # (num_envs, 3)

    # Check X bounds (lateral)
    x_out = torch.abs(local_pos[:, 0]) > x_limit

    # Check Y bounds (forward/backward from goal)
    y_too_far_forward = local_pos[:, 1] > y_limit_forward
    y_too_far_back = local_pos[:, 1] < y_limit_backward

    # Terminate if any boundary violated
    out_of_bounds = x_out | y_too_far_forward | y_too_far_back

    return out_of_bounds


def ball_on_ground_behind_robot(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.2,
    y_threshold: float = -1.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
) -> torch.Tensor:
    """Terminate if ball has settled on ground far behind robot.

    This indicates the kick is over and robot failed to block. We can
    reset the episode to train on new kicks.

    Args:
        env: Environment.
        height_threshold: Ball height threshold (meters) to consider "on ground".
        y_threshold: Y position threshold (meters) to consider "behind robot".
        robot_cfg: Robot configuration.
        ball_cfg: Ball configuration.

    Returns:
        Bool tensor of shape (num_envs,). True if ball settled behind, False otherwise.
    """
    ball: Entity = env.scene[ball_cfg.name]

    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)
    ball_vel = ball.data.root_link_lin_vel_w  # (num_envs, 3)
    env_origins = env.scene.env_origins  # (num_envs, 3)

    # Ball height and position relative to origin
    local_ball = ball_pos - env_origins
    ball_height = local_ball[:, 2]
    ball_y = local_ball[:, 1]

    # Ball velocity magnitude
    ball_speed = torch.norm(ball_vel, dim=-1)

    # Conditions:
    # 1. Ball is low (near ground)
    # 2. Ball is behind robot/goal (negative Y)
    # 3. Ball has stopped moving (settled)
    on_ground = ball_height < height_threshold
    behind_robot = ball_y < y_threshold
    stopped = ball_speed < 0.5  # Less than 0.5 m/s

    settled_behind = on_ground & behind_robot & stopped

    return settled_behind
