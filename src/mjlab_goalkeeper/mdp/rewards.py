"""Custom reward functions for goalkeeper environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ============================================================================
# Base Stability Rewards (Phase 1: Learn to Balance)
# ============================================================================


def upright_posture(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for maintaining upright posture (penalize tilt).

    Uses projected gravity to measure uprightness. When robot is perfectly
    upright, projected gravity is (0, 0, -1).

    Returns:
        Reward of shape (num_envs,). 1.0 when upright, decreases as robot tilts.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Projected gravity in body frame - should be close to (0, 0, -1) when upright
    proj_gravity = asset.data.projected_gravity_b  # (num_envs, 3)

    # Reward based on Z component being close to -1
    # When upright: proj_gravity[2] ≈ -1, so (proj_gravity[2] + 1) ≈ 0
    upright_reward = torch.exp(-torch.square(proj_gravity[:, 2] + 1.0) / 0.25)

    return upright_reward


def body_orientation_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Strong penalty for body tilt in pitch and roll (prevent leaning forward/sideways).

    This is stricter than upright_posture and specifically prevents the robot
    from leaning/falling while still getting position rewards.

    Returns:
        Penalty of shape (num_envs,). Negative when body tilts away from vertical.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Projected gravity should be (0, 0, -1) when perfectly upright
    proj_gravity = asset.data.projected_gravity_b  # (num_envs, 3)

    # Penalize ANY deviation in X and Y (pitch and roll)
    # When upright: proj_gravity[0] ≈ 0 and proj_gravity[1] ≈ 0
    xy_tilt = torch.sum(torch.square(proj_gravity[:, :2]), dim=1)

    # Return negative penalty (so it penalizes tilt)
    return -xy_tilt * 10.0  # Scale up for strong penalty


def feet_contact_reward(
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for keeping both feet on the ground (encourage stability).

    Args:
        env: Environment.
        sensor_names: List of foot contact sensor names.
        asset_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when both feet in contact, 0.5 for one foot.
    """
    asset: Entity = env.scene[asset_cfg.name]

    contact_list = []
    for sensor_name in sensor_names:
        sensor_data = asset.data.sensor_data[sensor_name]
        foot_contact = (sensor_data[:, 0] > 0).float()  # Binary: 1.0 if contact
        contact_list.append(foot_contact)

    contacts = torch.stack(contact_list, dim=1)  # (num_envs, num_feet)

    # Reward proportional to number of feet in contact
    num_feet_contact = torch.sum(contacts, dim=1)  # (num_envs,)

    # Normalize by number of feet (2 feet = 1.0 reward)
    return num_feet_contact / len(sensor_names)


def base_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.75,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for maintaining standing height (avoid crouching/falling).

    This is MUCH more robust than joint posture for preventing crouching.
    Directly measures what we care about: standing height.

    Args:
        env: Environment.
        target_height: Target base height in meters (G1 standing ~0.75m).
        std: Standard deviation for exponential kernel.
        asset_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 at target height, decreases as height deviates.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Work in LOCAL coordinates
    env_origins = env.scene.env_origins  # (num_envs, 3)
    base_pos_w = asset.data.root_link_pos_w  # (num_envs, 3)

    # Height relative to environment origin
    height = base_pos_w[:, 2] - env_origins[:, 2]  # (num_envs,)

    # Exponential reward centered at target height
    height_error = torch.square(height - target_height)
    return torch.exp(-height_error / (std**2))


# ============================================================================
# Movement Rewards (Phase 2: Learn to Move to Ball)
# ============================================================================


def move_to_ball_x_position(
    env: ManagerBasedRlEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
) -> torch.Tensor:
    """Reward for aligning X position with ball's predicted goal-line crossing.

    Predicts where the ball will cross the goal line (Y coordinate) and rewards
    robot for moving to that X position. Only rewards lateral (X) movement.

    Args:
        env: Environment.
        std: Standard deviation for exponential kernel (smaller = stricter).
        robot_cfg: Robot configuration.
        ball_cfg: Ball configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when perfectly aligned, decreases with distance.
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    # Get positions
    robot_pos = robot.data.root_link_pos_w  # (num_envs, 3)
    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)
    ball_vel = ball.data.root_link_lin_vel_w  # (num_envs, 3)

    # Goal line Y position (where ball needs to be intercepted)
    goal_line_y = -0.5  # From goal entity configuration

    # Predict where ball will cross goal line
    # time_to_goal = (goal_line_y - ball_y) / ball_vel_y
    # target_x = ball_x + ball_vel_x * time_to_goal

    dy = goal_line_y - ball_pos[:, 1]  # Distance to goal line

    # Avoid division by zero - if ball not moving toward goal, use current X position
    ball_vel_y = ball_vel[:, 1].clamp(max=-0.1)  # Only consider movement toward goal
    time_to_goal = dy / (ball_vel_y - 1e-8)  # Time until ball reaches goal line

    # Predicted X position where ball crosses goal line
    predicted_ball_x = ball_pos[:, 0] + ball_vel[:, 0] * time_to_goal

    # Clamp predicted position to goal width (avoid chasing balls way off target)
    goal_width = 3.5  # From goal entity
    predicted_ball_x = torch.clamp(predicted_ball_x, -goal_width / 2, goal_width / 2)

    # Error between robot X and predicted ball X
    x_error = torch.square(robot_pos[:, 0] - predicted_ball_x)

    # Exponential reward
    return torch.exp(-x_error / (std**2))


def front_facing_reward(
    env: ManagerBasedRlEnv,
    target_yaw: float = math.pi / 2,  # 90 degrees = facing +Y (toward ball)
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for facing the ball (yaw aligned with target direction).

    Args:
        env: Environment.
        target_yaw: Target yaw angle in radians (π/2 = facing +Y).
        std: Standard deviation for exponential kernel.
        asset_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when facing target direction.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get robot yaw (heading in world frame)
    robot_yaw = asset.data.heading_w  # (num_envs,)

    # Compute angular error (wrap to [-π, π])
    yaw_error = robot_yaw - target_yaw
    yaw_error = torch.atan2(
        torch.sin(yaw_error), torch.cos(yaw_error)
    )  # Wrap to [-π, π]

    # Exponential reward
    return torch.exp(-torch.square(yaw_error) / (std**2))


def lateral_velocity_reward(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
) -> torch.Tensor:
    """Reward for moving laterally (X direction) toward ball's predicted position.

    Encourages walking/stepping to intercept rather than standing still.
    Only rewards X velocity, penalizes Y velocity (don't move toward/away from goal).

    Returns:
        Reward of shape (num_envs,). Positive when moving correctly.
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    robot_pos = robot.data.root_link_pos_w[:, 0]  # X position only
    ball_pos = ball.data.root_link_pos_w[:, 0]  # X position only

    # Desired direction: sign of (ball_x - robot_x)
    desired_direction = torch.sign(ball_pos - robot_pos)  # +1 or -1

    # Robot's X velocity (in world frame)
    robot_vel_x = robot.data.root_link_lin_vel_w[:, 0]

    # Reward velocity in the correct direction
    # If robot should move right (+X) and is moving right, reward is positive
    directional_velocity = desired_direction * robot_vel_x

    # Only reward positive velocities (moving in correct direction)
    # Use tanh to saturate at reasonable velocities (~0.5 m/s)
    return torch.tanh(torch.clamp(directional_velocity, min=0.0) / 0.5)


def track_lateral_velocity_to_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    max_velocity: float,
    kp: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track lateral velocity to command target X position (like track_lin_vel_exp).

    Uses command's precomputed interception point. Simple proportional control:
    desired_vel_x = kp * position_error

    ONLY X-axis velocity. Y and Z must stay zero.

    Args:
        env: Environment.
        command_name: Name of penalty kick command.
        std: Standard deviation for exponential kernel.
        max_velocity: Maximum desired velocity (m/s).
        kp: Proportional gain.
        robot_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when perfectly tracking velocity.
    """
    robot: Entity = env.scene[robot_cfg.name]

    # Get target X from command (in LOCAL coordinates)
    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        return torch.zeros(env.num_envs, device=env.device)

    target_x_local = command_term.interception_x_local  # (num_envs,)

    # Get robot X in LOCAL coordinates
    env_origins = env.scene.env_origins
    robot_pos_w = robot.data.root_link_pos_w  # (num_envs, 3)
    robot_x_local = robot_pos_w[:, 0] - env_origins[:, 0]

    # Position error
    x_error = target_x_local - robot_x_local  # (num_envs,)

    # Desired lateral velocity (proportional control)
    desired_vel_x = (kp * x_error).clamp(-max_velocity, max_velocity)

    # Get actual velocity (world frame)
    actual_vel = robot.data.root_link_lin_vel_w  # (num_envs, 3)

    # Build desired velocity: X only, Y and Z must be zero
    desired_vel = torch.zeros_like(actual_vel)
    desired_vel[:, 0] = desired_vel_x
    desired_vel[:, 1] = 0.0  # No forward/back
    desired_vel[:, 2] = 0.0  # No jump

    # Velocity tracking error (all axes contribute)
    vel_error = torch.sum(torch.square(desired_vel - actual_vel), dim=1)

    return torch.exp(-vel_error / (std**2))


def position_to_target_x(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for BEING at command's target X position (teaches WHERE to stop).

    ONLY considers X-axis position (lateral alignment).

    Args:
        env: Environment.
        command_name: Name of penalty kick command.
        std: Standard deviation for exponential kernel.
        robot_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when at target, decreases with distance.
    """
    robot: Entity = env.scene[robot_cfg.name]

    # Get target X from command (in LOCAL coordinates)
    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        return torch.zeros(env.num_envs, device=env.device)

    target_x_local = command_term.interception_x_local  # (num_envs,)

    # Get robot X in LOCAL coordinates
    env_origins = env.scene.env_origins
    robot_pos_w = robot.data.root_link_pos_w
    robot_x_local = robot_pos_w[:, 0] - env_origins[:, 0]

    # Position error (absolute)
    x_error = torch.abs(target_x_local - robot_x_local)

    return torch.exp(-torch.square(x_error) / (std**2))


def stay_on_goal_line_y(
    env: ManagerBasedRlEnv,
    target_y: float = -0.3,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for staying at goal line Y position (don't drift forward/backward).

    Robot should stay at y=-0.3m (spawn position, in front of goal).
    This prevents drifting toward the ball or backing into the goal.

    Args:
        env: Environment.
        target_y: Target Y position in LOCAL coordinates (default: -0.3m).
        std: Standard deviation for exponential kernel.
        asset_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when on goal line, decreases with Y deviation.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Work in LOCAL coordinates
    env_origins = env.scene.env_origins  # (num_envs, 3)
    robot_pos_w = asset.data.root_link_pos_w  # (num_envs, 3)
    robot_pos = robot_pos_w - env_origins  # Local position

    # Y position error (how far from goal line)
    y_error = torch.abs(robot_pos[:, 1] - target_y)  # (num_envs,)

    # Exponential reward: peaks at target Y position
    return torch.exp(-torch.square(y_error) / (std**2))


# ============================================================================
# Hand Positioning Rewards (Phase 3: Learn to Block)
# ============================================================================


def hand_to_ball_distance(
    env: ManagerBasedRlEnv,
    std: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
) -> torch.Tensor:
    """Reward for positioning hands close to the ball.

    Uses minimum distance between either hand and ball. This encourages
    the robot to raise hands and position them for blocking.

    Args:
        env: Environment.
        std: Standard deviation for exponential kernel.
        robot_cfg: Robot configuration.
        ball_cfg: Ball configuration.

    Returns:
        Reward of shape (num_envs,). 1.0 when hand touches ball, decreases with distance.
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    # Get hand positions (from observations)
    # Find hand body indices
    left_hand_name = "left_wrist_yaw_link"
    right_hand_name = "right_wrist_yaw_link"

    body_names = robot.body_names
    left_idx = body_names.index(left_hand_name)
    right_idx = body_names.index(right_hand_name)

    left_hand_pos = robot.data.body_link_pos_w[:, left_idx, :]  # (num_envs, 3)
    right_hand_pos = robot.data.body_link_pos_w[:, right_idx, :]  # (num_envs, 3)

    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)

    # Distance from each hand to ball
    left_dist = torch.norm(left_hand_pos - ball_pos, dim=-1)  # (num_envs,)
    right_dist = torch.norm(right_hand_pos - ball_pos, dim=-1)  # (num_envs,)

    # Use minimum distance (either hand can block)
    min_dist = torch.min(left_dist, right_dist)

    # Exponential reward (peaks when hand touches ball)
    return torch.exp(-torch.square(min_dist) / (std**2))


def ball_contact_reward(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Large reward for making contact with the ball.

    This is a binary reward that triggers when robot successfully touches ball.

    Args:
        env: Environment.
        robot_cfg: Robot configuration.
        ball_cfg: Ball configuration.
        threshold: Distance threshold for contact detection.

    Returns:
        Reward of shape (num_envs,). 1.0 if contact detected, 0.0 otherwise.
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    robot_pos = robot.data.root_link_pos_w  # (num_envs, 3)
    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)

    # Distance between robot base and ball
    distance = torch.norm(ball_pos - robot_pos, dim=-1)  # (num_envs,)

    # Binary contact detection
    contact = (distance < threshold).float()

    return contact


def goal_prevention_penalty(
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    """Large penalty if the ball scores a goal.

    This is a strong negative signal to encourage blocking behavior.

    Returns:
        Penalty of shape (num_envs,). -1.0 if goal scored, 0.0 otherwise.
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

    goal_scored = (x_in_goal & y_behind_line & y_in_goal & z_in_goal).float()

    # Return negative reward (penalty)
    return -goal_scored


# ============================================================================
# Regularization Rewards (Smoothness & Efficiency)
# ============================================================================


def action_smoothness(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalize large changes in actions (encourage smooth control).

    Note: This is similar to action_rate_l2 from mjlab, kept here for reference.
    Consider using mjlab_mdp.action_rate_l2 instead.

    Returns:
        Penalty of shape (num_envs,). Negative value based on action changes.
    """
    # Get current and previous actions
    if len(env.action_manager.action_history) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    current_action = env.action_manager.action_history[-1]
    previous_action = env.action_manager.action_history[-2]

    # L2 norm of action difference
    action_diff = torch.sum(torch.square(current_action - previous_action), dim=-1)

    return -action_diff


def energy_expenditure(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalize excessive torques (encourage energy efficiency).

    This encourages smooth, efficient movements rather than violent actions.

    Returns:
        Penalty of shape (num_envs,). Negative value based on torque magnitude.
    """
    # Get joint torques (controls)
    torques = env.sim.data.ctrl  # (num_envs, num_actuators)

    # L2 norm of torques
    torque_magnitude = torch.sum(torch.square(torques), dim=-1)

    return -torque_magnitude * 0.0001  # Scale down to avoid overwhelming other rewards


# ============================================================================
# Posture & Joint Limit Rewards (from velocity task)
# ============================================================================


def posture_reward(
    env: ManagerBasedRlEnv,
    std: dict[str, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for maintaining default joint posture.

    This is adapted from the velocity task to encourage natural standing pose.

    Args:
        env: Environment.
        std: Dictionary mapping joint name patterns to standard deviations.
        asset_cfg: Robot configuration.

    Returns:
        Reward of shape (num_envs,).
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get current and default joint positions
    current_pos = asset.data.joint_pos  # (num_envs, num_joints)
    default_pos = asset.data.default_joint_pos  # (num_envs, num_joints)

    # Compute per-joint errors with different std for each joint
    joint_names = asset.joint_names
    std_values = torch.ones(len(joint_names), device=env.device)

    # Apply std from config (pattern matching)
    import re

    for pattern, std_value in std.items():
        for i, name in enumerate(joint_names):
            if re.match(pattern, name):
                std_values[i] = std_value

    # Compute weighted squared error
    errors = torch.square((current_pos - default_pos) / std_values.unsqueeze(0))

    # Sum and apply exponential
    total_error = torch.sum(errors, dim=-1)
    return torch.exp(-total_error)


def joint_limits_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize approaching joint position limits.

    Returns:
        Penalty of shape (num_envs,). Negative when near limits.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get joint positions and limits
    joint_pos = asset.data.joint_pos  # (num_envs, num_joints)

    # Access joint limits from MuJoCo model
    # joint_range is (num_joints, 2) where [:, 0] is lower, [:, 1] is upper
    joint_ids = asset.indexing.joint_ids.cpu().numpy()  # Convert to numpy for indexing
    joint_range = env.sim.mj_model.jnt_range[joint_ids]  # (num_joints, 2)

    lower_limits = torch.tensor(joint_range[:, 0], device=env.device)  # (num_joints,)
    upper_limits = torch.tensor(joint_range[:, 1], device=env.device)  # (num_joints,)

    # Compute distance to limits (normalized by range)
    range_size = upper_limits - lower_limits
    lower_violation = torch.clamp((lower_limits - joint_pos) / range_size, min=0.0)
    upper_violation = torch.clamp((joint_pos - upper_limits) / range_size, min=0.0)

    # Sum violations across joints
    total_violation = torch.sum(lower_violation + upper_violation, dim=-1)

    return -total_violation
