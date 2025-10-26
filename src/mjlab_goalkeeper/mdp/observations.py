"""Custom observation functions for goalkeeper environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv


def entity_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the position of an entity in world frame.

    Args:
        env: The environment.
        asset_cfg: Configuration for the entity to observe.

    Returns:
        Position tensor of shape (num_envs, 3).
    """
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.root_link_pos_w


def entity_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the linear velocity of an entity in world frame.

    Args:
        env: The environment.
        asset_cfg: Configuration for the entity to observe.

    Returns:
        Linear velocity tensor of shape (num_envs, 3).
    """
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.root_link_lin_vel_w


def entity_relative_position(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    reference_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get the relative position between two entities.

    Args:
        env: The environment.
        target_asset_cfg: Configuration for the target entity.
        reference_asset_cfg: Configuration for the reference entity.

    Returns:
        Relative position tensor of shape (num_envs, 3).
    """
    target: Entity = env.scene[target_asset_cfg.name]
    reference: Entity = env.scene[reference_asset_cfg.name]

    target_pos = target.data.root_link_pos_w
    reference_pos = reference.data.root_link_pos_w

    return target_pos - reference_pos


def entity_contact_force(
    env: ManagerBasedEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get contact force magnitude from a contact sensor.

    Args:
        env: The environment.
        sensor_name: Name of the contact sensor.
        asset_cfg: Configuration for the entity with the sensor.

    Returns:
        Contact force tensor of shape (num_envs,). Returns 1.0 if contact
        is detected, 0.0 otherwise (when using data=("found",)).
    """
    entity: Entity = env.scene[asset_cfg.name]

    if sensor_name not in entity.sensor_names:
        raise ValueError(
            f"Sensor '{sensor_name}' not found in entity '{asset_cfg.name}'. "
            f"Available sensors: {entity.sensor_names}"
        )

    # With data=("found",) and reduce="netforce", the sensor returns 1.0 if
    # any contact is detected, 0.0 otherwise
    contact_data = entity.data.sensor_data[sensor_name]

    # Return the first element (found flag)
    return contact_data[..., 0]


def entity_contact_detected(
    env: ManagerBasedEnv,
    entity1_cfg: SceneEntityCfg,
    entity2_cfg: SceneEntityCfg,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Detect if entities are in close proximity (proxy for contact).

    This is the proper way to detect inter-entity "contacts" in mjlab.
    ContactSensorCfg only works for intra-entity contacts or entity-terrain contacts.

    For true contact detection between entities, use proximity-based approach:
    distance between entities < threshold indicates contact/near-contact.

    Args:
        env: The environment.
        entity1_cfg: Configuration for the first entity.
        entity2_cfg: Configuration for the second entity.
        threshold: Distance threshold in meters for contact detection.

    Returns:
        Binary tensor of shape (num_envs,). 1.0 if within threshold, 0.0 otherwise.
    """
    entity1: Entity = env.scene[entity1_cfg.name]
    entity2: Entity = env.scene[entity2_cfg.name]

    # Get positions of both entities
    pos1 = entity1.data.root_link_pos_w  # (num_envs, 3)
    pos2 = entity2.data.root_link_pos_w  # (num_envs, 3)

    # Calculate distance between entities
    distance = torch.norm(pos2 - pos1, dim=-1)  # (num_envs,)

    # Contact detected if distance below threshold
    contact_detected = (distance < threshold).float()

    return contact_detected


def goal_scored_detection(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> torch.Tensor:
    """Detect if the ball has scored a goal.

    A goal is scored when the ball crosses the goal line (within goal boundaries).

    Goal dimensions:
    - Width: 7.32m (±3.66m from center)
    - Height: 2.44m (0 to 2.44m)
    - Goal line: y = -0.5m (goal extends back to y = -1.5m)

    Args:
        env: The environment.
        ball_cfg: Configuration for the ball entity.
        goal_cfg: Configuration for the goal entity.

    Returns:
        Binary tensor of shape (num_envs, 1). 1.0 if goal scored, 0.0 otherwise.
    """
    ball: Entity = env.scene[ball_cfg.name]
    goal: Entity = env.scene[goal_cfg.name]

    # Get ball position
    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)

    # Get goal position (center of goal)
    goal_pos = goal.data.root_link_pos_w  # (num_envs, 3)

    # Goal dimensions (smaller, more appropriate for humanoid robot)
    goal_width = 3.5  # meters (reduced from 7.32m FIFA standard)
    goal_height = 2.0  # meters (reduced from 2.44m)
    goal_line_y = -0.5  # Goal line position
    goal_depth = 1.0  # How far back the goal extends

    # Check if ball is within goal boundaries
    # X: within ±goal_width/2 of goal center
    x_in_goal = torch.abs(ball_pos[:, 0] - goal_pos[:, 0]) < (goal_width / 2)

    # Y: behind goal line (negative Y)
    y_behind_line = ball_pos[:, 1] < goal_line_y

    # Y: not too far back (within goal depth)
    y_in_goal = ball_pos[:, 1] > (goal_line_y - goal_depth)

    # Z: within goal height (above ground, below crossbar)
    z_in_goal = (ball_pos[:, 2] > 0.0) & (ball_pos[:, 2] < goal_height)

    # Goal scored if all conditions met
    goal_scored = (x_in_goal & y_behind_line & y_in_goal & z_in_goal).float()

    return goal_scored.unsqueeze(-1)  # (num_envs, 1)


def robot_ball_contact(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("football_ball"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Detect contact between robot and ball.

    This is a convenience wrapper around entity_contact_detected specifically
    for robot-ball interactions.

    Args:
        env: The environment.
        robot_cfg: Configuration for the robot entity.
        ball_cfg: Configuration for the ball entity.
        threshold: Distance threshold in meters for contact detection.

    Returns:
        Binary tensor of shape (num_envs, 1). 1.0 if contact, 0.0 otherwise.
    """
    contact = entity_contact_detected(env, robot_cfg, ball_cfg, threshold=threshold)
    return contact.unsqueeze(-1)  # (num_envs, 1)


def hand_position(
    env: ManagerBasedEnv,
    hand_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the position of a robot hand in world frame.

    Args:
        env: The environment.
        hand_name: Name of the hand body (e.g., "left_hand_roll_link", "right_hand_roll_link").
        asset_cfg: Configuration for the robot entity.

    Returns:
        Position tensor of shape (num_envs, 3).
    """
    robot: Entity = env.scene[asset_cfg.name]

    # Find the body index for the hand
    body_names = robot.body_names
    if hand_name not in body_names:
        raise ValueError(
            f"Hand '{hand_name}' not found in robot. Available bodies: {body_names}"
        )

    body_idx = body_names.index(hand_name)

    # Get hand position in world frame
    hand_pos = robot.data.body_link_pos_w[:, body_idx, :]  # (num_envs, 3)

    return hand_pos
