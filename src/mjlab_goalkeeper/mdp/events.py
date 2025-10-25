"""Custom event functions for goalkeeper environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv


def reset_robot_on_goal_line(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_std: float = 0.5,
    yaw_range: tuple[float, float] = (-0.3, 0.3),
) -> None:
    """Reset robot in front of goal with Gaussian distribution for X position.

    The robot spawns in front of the goal, ready to defend:
    - X position: Gaussian distribution centered at origin (std = x_std)
    - Y position: Slightly in front of goal line (0.2m forward from goal)
    - Z position: Uses robot's default standing height
    - Yaw: Facing toward penalty spot (positive Y) with small variation

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        asset_cfg: Configuration for the robot asset.
        x_std: Standard deviation for Gaussian X position (default: 0.5m).
        yaw_range: Range for yaw randomization around forward direction (default: ±0.3 rad).
    """
    robot: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Sample X position with Gaussian distribution (centered at 0)
    x_pos = torch.randn(num_envs, device=env.device) * x_std

    # Y position: In front of goal line (goal is at -0.5m, robot at -0.3m)
    # This puts robot 0.2m in front of goal, clear of goal geometry
    y_pos = torch.full((num_envs,), -0.3, device=env.device)

    # Z position: Use robot's default standing height (from init state)
    # This prevents spawning inside the ground
    default_z = robot.data.default_root_state[
        0, 2
    ].item()  # Get default Z from first env
    z_pos = torch.full((num_envs,), default_z, device=env.device)

    # Combine positions
    positions = torch.stack([x_pos, y_pos, z_pos], dim=-1)
    positions += env.scene.env_origins[env_ids]

    # Yaw: Face toward penalty spot (positive Y direction) with small random variation
    # yaw=π/2 (90°) means facing positive Y (toward the ball)
    base_yaw = torch.pi / 2  # 90 degrees - facing toward positive Y
    yaw = base_yaw + torch.empty(num_envs, device=env.device).uniform_(*yaw_range)

    # Convert yaw to quaternion (rotation around Z axis)
    quat_w = torch.cos(yaw / 2)
    quat_x = torch.zeros_like(yaw)
    quat_y = torch.zeros_like(yaw)
    quat_z = torch.sin(yaw / 2)
    orientations = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=-1)

    # Zero velocities
    lin_vel = torch.zeros(num_envs, 3, device=env.device)
    ang_vel = torch.zeros(num_envs, 3, device=env.device)
    velocities = torch.cat([lin_vel, ang_vel], dim=-1)

    # Combine into root state [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    root_state = torch.cat([positions, orientations, velocities], dim=-1)

    # Write to simulation
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)


def update_ball_circular_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    radius: float = 10.0,
    angular_velocity: float = 0.3,
    height: float = 0.11,
) -> None:
    """Update ball velocity to maintain smooth circular motion around the robot.

    This event continuously updates the ball's velocity to move it in a circle
    around the robot's origin, creating realistic goalkeeper training scenarios.

    The ball moves tangentially at each point on the circle. The velocity is
    recomputed based on current position to maintain smooth circular trajectory.

    Args:
        env: The environment.
        env_ids: Environment IDs to apply the event to. If None, applies to all.
        asset_cfg: Configuration for the ball asset.
        radius: Radius of the circular path in meters.
        angular_velocity: Angular velocity in rad/s (positive = counterclockwise).
        height: Height of the ball above ground (ball radius for ground contact).
    """
    ball: Entity = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Get current ball position
    current_pos = ball.data.root_link_pos_w[env_ids]  # (num_envs, 3)

    # Get environment origins to work in local coordinates
    env_origins = env.scene.env_origins[env_ids]  # (num_envs, 3)
    local_pos = current_pos - env_origins  # Position relative to env origin

    # Current position on XY plane (ignore Z)
    x = local_pos[:, 0]
    y = local_pos[:, 1]

    # Current angle on the circle
    current_radius = torch.sqrt(x**2 + y**2)

    # Tangent velocity for circular motion: v_tangent = ω × r
    # At position (x, y), tangent direction is (-y, x) normalized
    # Then scaled by v = ω * R
    linear_speed = angular_velocity * radius  # m/s

    # Tangent direction (perpendicular to radial direction)
    tangent_x = -y / (current_radius + 1e-8)  # Avoid division by zero
    tangent_y = x / (current_radius + 1e-8)

    # Tangent velocity
    vel_x = tangent_x * linear_speed
    vel_y = tangent_y * linear_speed
    vel_z = torch.zeros_like(vel_x)

    # Combine linear and angular velocity
    lin_vel = torch.stack([vel_x, vel_y, vel_z], dim=-1)  # (num_envs, 3)
    ang_vel = torch.zeros_like(lin_vel)  # No rotation

    # Write velocity to simulation
    velocity = torch.cat([lin_vel, ang_vel], dim=-1)  # (num_envs, 6)
    ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def reset_ball_on_circle(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    radius: float = 10.0,
    angular_velocity: float = 0.3,
    height: float = 0.11,
) -> None:
    """Reset ball to a position on the circular path with appropriate velocity.

    Places the ball on a circle around the robot origin and gives it initial
    tangential velocity for smooth circular motion.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        asset_cfg: Configuration for the ball asset.
        radius: Radius of the circular path in meters.
        angular_velocity: Angular velocity in rad/s.
        height: Height of the ball above ground (ball radius for ground contact).
    """
    ball: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Place ball at a fixed starting position on the circle
    # Start at angle 0 (positive X axis)
    angle = torch.zeros(num_envs, device=env.device)

    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    z = torch.full((num_envs,), height, device=env.device)

    # Position in world frame
    positions = torch.stack([x, y, z], dim=-1)
    positions += env.scene.env_origins[env_ids]

    # Identity quaternion (no rotation)
    orientations = torch.zeros(num_envs, 4, device=env.device)
    orientations[:, 0] = 1.0  # w component

    # Initial tangential velocity for circular motion
    linear_speed = angular_velocity * radius
    vel_x = -torch.sin(angle) * linear_speed  # Tangent direction at angle
    vel_y = torch.cos(angle) * linear_speed
    vel_z = torch.zeros_like(vel_x)

    lin_vel = torch.stack([vel_x, vel_y, vel_z], dim=-1)
    ang_vel = torch.zeros_like(lin_vel)
    velocities = torch.cat([lin_vel, ang_vel], dim=-1)

    # Combine into root state [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    root_state = torch.cat([positions, orientations, velocities], dim=-1)

    # Write to simulation
    ball.write_root_state_to_sim(root_state, env_ids=env_ids)
