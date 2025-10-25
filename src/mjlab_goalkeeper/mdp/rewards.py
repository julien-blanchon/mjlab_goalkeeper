from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    desired = torch.zeros_like(actual)
    desired[:, :2] = command[:, :2]
    lin_vel_error = torch.sum(torch.square(desired - actual), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    desired = torch.zeros_like(actual)
    desired[:, 2] = command[:, 2]
    ang_vel_error = torch.sum(torch.square(desired - actual), dim=1)
    return torch.exp(-ang_vel_error / std**2)


class feet_air_time:
    """Reward long steps taken by the feet.

    This rewards the agent for lifting feet off the ground for longer than a threshold.
    Provides continuous reward signal during flight phase and smooth command scaling.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.threshold_min = cfg.params["threshold_min"]
        self.threshold_max = cfg.params.get("threshold_max", self.threshold_min + 0.3)
        self.asset_name = cfg.params["asset_name"]
        self.sensor_names = cfg.params["sensor_names"]
        self.num_feet = len(self.sensor_names)
        self.command_name = cfg.params["command_name"]
        self.command_threshold = cfg.params["command_threshold"]
        self.reward_mode = cfg.params.get("reward_mode", "continuous")
        self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
        self.command_scale_width = cfg.params.get("command_scale_width", 0.2)

        asset: Entity = env.scene[self.asset_name]
        for sensor_name in self.sensor_names:
            if sensor_name not in asset.sensor_names:
                raise ValueError(
                    f"Sensor '{sensor_name}' not found in asset '{self.asset_name}'"
                )

        self.current_air_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )
        self.current_contact_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )
        self.last_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        asset: Entity = env.scene[self.asset_name]

        contact_list = []
        for sensor_name in self.sensor_names:
            sensor_data = asset.data.sensor_data[sensor_name]
            foot_contact = sensor_data[:, 0] > 0
            contact_list.append(foot_contact)

        in_contact = torch.stack(contact_list, dim=1)
        in_air = ~in_contact

        # Detect first contact (landing).
        first_contact = (self.current_air_time > 0) & in_contact

        # Save air time when landing.
        self.last_air_time = torch.where(
            first_contact, self.current_air_time, self.last_air_time
        )

        # Update air time and contact time.
        self.current_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.current_air_time),  # Reset when in contact.
            self.current_air_time + env.step_dt,  # Increment when in air.
        )

        self.current_contact_time = torch.where(
            in_contact,
            self.current_contact_time + env.step_dt,  # Increment when in contact.
            torch.zeros_like(self.current_contact_time),  # Reset when in air.
        )

        if self.reward_mode == "continuous":
            # Give constant reward of 1.0 for each foot that's in air and above threshold.
            exceeds_min = self.current_air_time > self.threshold_min
            below_max = self.current_air_time <= self.threshold_max
            reward_per_foot = torch.where(
                in_air & exceeds_min & below_max,
                torch.ones_like(self.current_air_time),
                torch.zeros_like(self.current_air_time),
            )
            reward = torch.sum(reward_per_foot, dim=1)
        else:
            # This mode gives (air_time - threshold) as reward on landing.
            air_time_over_min = (self.last_air_time - self.threshold_min).clamp(min=0.0)
            air_time_clamped = air_time_over_min.clamp(
                max=self.threshold_max - self.threshold_min
            )
            reward = torch.sum(air_time_clamped * first_contact, dim=1) / env.step_dt

        command = env.command_manager.get_command(self.command_name)
        assert command is not None
        command_norm = torch.norm(command[:, :2], dim=1)
        if self.command_scale_type == "smooth":
            scale = 0.5 * (
                1.0
                + torch.tanh(
                    (command_norm - self.command_threshold) / self.command_scale_width
                )
            )
            reward *= scale
        else:
            reward *= command_norm > self.command_threshold
        return reward

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0


def foot_clearance_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_list = []
    for sensor_name in sensor_names:
        sensor_data = asset.data.sensor_data[sensor_name]
        foot_contact = sensor_data[:, 0] > 0
        contact_list.append(foot_contact)
    contacts = torch.stack(contact_list, dim=1)
    geom_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
    return torch.sum(geom_vel.norm(dim=-1) * contacts, dim=1)


def hand_to_ball_distance(
    env: ManagerBasedRlEnv,
    std: float = 0.3,
    left_hand_body: str = "left_hand_roll_link",
    right_hand_body: str = "right_hand_roll_link",
) -> torch.Tensor:
    """Reward for getting hands close to the ball (exponential kernel).

    Encourages the robot to reach for the ball with its hands rather than
    just blocking with its body.

    Args:
      env: The environment.
      std: Standard deviation for exponential kernel.
      left_hand_body: Name of the left hand body.
      right_hand_body: Name of the right hand body.

    Returns:
      Reward tensor of shape (num_envs,).
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["football_ball"]

    # Get hand positions
    left_hand_idx = robot.body_names.index(left_hand_body)
    right_hand_idx = robot.body_names.index(right_hand_body)

    left_hand_pos = robot.data.body_link_pos_w[:, left_hand_idx, :]  # (num_envs, 3)
    right_hand_pos = robot.data.body_link_pos_w[:, right_hand_idx, :]  # (num_envs, 3)

    # Get ball position
    ball_pos = ball.data.root_link_pos_w  # (num_envs, 3)

    # Distance from each hand to ball
    left_dist = torch.norm(ball_pos - left_hand_pos, dim=-1)  # (num_envs,)
    right_dist = torch.norm(ball_pos - right_hand_pos, dim=-1)  # (num_envs,)

    # Use the closer hand (encourages using either hand)
    min_dist = torch.min(left_dist, right_dist)

    # Exponential reward (peaks when distance = 0)
    return torch.exp(-min_dist / std**2)


class post_contact_stabilization:
    """Reward for staying stable after touching the ball.

    This encourages the robot to maintain balance and recover after making a save,
    rather than falling over after blocking the ball.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.std = cfg.params["std"]
        self.contact_memory_steps = cfg.params.get("contact_memory_steps", 50)
        self.contact_threshold = cfg.params.get("contact_threshold", 0.5)
        self.nominal_height = cfg.params.get("nominal_height", 0.76)

        # Track steps since last ball contact
        self.steps_since_contact = torch.full(
            (env.num_envs,), self.contact_memory_steps + 1, device=env.device
        )

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        robot: Entity = env.scene["robot"]
        ball: Entity = env.scene["football_ball"]

        # Check if robot is currently in contact with ball
        distance = torch.norm(
            ball.data.root_link_pos_w - robot.data.root_link_pos_w, dim=-1
        )
        in_contact = distance < self.contact_threshold

        # Update steps since contact
        self.steps_since_contact = torch.where(
            in_contact,
            torch.zeros_like(self.steps_since_contact),  # Reset to 0 on contact
            self.steps_since_contact + 1,  # Increment otherwise
        )

        # Only reward stabilization within memory window after contact
        recently_contacted = self.steps_since_contact < self.contact_memory_steps

        # Compute stability error
        # 1. Height error (maintain standing height)
        height_error = torch.abs(robot.data.root_link_pos_w[:, 2] - self.nominal_height)

        # 2. Angular velocity (low spinning)
        ang_vel_mag = torch.norm(robot.data.root_link_ang_vel_b, dim=-1)

        # 3. Tilt (projected gravity should be [0, 0, -1] when upright)
        projected_grav = robot.data.projected_gravity_b
        tilt_error = torch.norm(
            projected_grav[:, :2], dim=-1
        )  # XY components should be ~0

        # Combine errors
        total_error = height_error + ang_vel_mag + tilt_error

        # Exponential reward (higher when more stable)
        stability_reward = torch.exp(-total_error / self.std**2)

        # Only apply reward if recently contacted ball
        reward = stability_reward * recently_contacted.float()

        return reward

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.steps_since_contact[env_ids] = self.contact_memory_steps + 1


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant reward for staying alive (upright) each step.

    This provides a continuous incentive to maintain balance and not fall over.
    Combined with fell_over termination, this creates a strong penalty for falling:
    - While upright: Gets +1.0 every step
    - When falls: Episode ends, no more alive rewards

    This prevents the "instant fall" problem during early training.

    Args:
      env: The environment.

    Returns:
      Constant reward of 1.0 for each step (num_envs,).
    """
    return torch.ones(env.num_envs, device=env.device)


class goal_prevented_by_robot:
    """Reward for actively preventing a goal (robot touched ball AND goal not scored).

    This ensures the robot only gets rewarded when it ACTIVELY saves a goal,
    not when the ball simply misses the target or hits the post.

    The reward is continuous after a successful save (robot touched + no goal).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.contact_threshold = cfg.params.get("contact_threshold", 0.5)

        # Track if robot has touched ball this episode
        self.has_touched_ball = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        robot: Entity = env.scene["robot"]
        ball: Entity = env.scene["football_ball"]

        # Check current contact
        distance = torch.norm(
            ball.data.root_link_pos_w - robot.data.root_link_pos_w, dim=-1
        )
        currently_touching = distance < self.contact_threshold

        # Update touch memory (once touched, stays true for the episode)
        self.has_touched_ball |= currently_touching

        # Check if goal was scored (import at runtime to avoid circular dependency)
        from mjlab_goalkeeper.mdp.observations import goal_scored_detection

        goal_scored = goal_scored_detection(env).squeeze(-1) > 0.5

        # Reward only if:
        # 1. Robot touched ball at some point (active intervention)
        # 2. Goal was NOT scored (successful save)
        # Give continuous reward after successful save
        successful_save = self.has_touched_ball & (~goal_scored)

        return successful_save.float()

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.has_touched_ball[env_ids] = False
