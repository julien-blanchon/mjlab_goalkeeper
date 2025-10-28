"""Custom command terms for goalkeeper environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class PenaltyKickCommand(CommandTerm):
    """Command term that shoots penalty kicks toward the goal.

    This command periodically resets the ball to the penalty spot and kicks it
    toward the goal with realistic ballistic trajectory, creating goalkeeper
    training scenarios.
    """

    cfg: PenaltyKickCommandCfg

    def __init__(self, cfg: PenaltyKickCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.robot_name]
        self.football: Entity = env.scene[cfg.football_name]
        self.goal: Entity = env.scene[cfg.goal_name]

        # Store current kick parameters for each environment
        self.kick_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_start_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Store interception point (where robot should move to) in LOCAL coordinates
        self.interception_x_local = torch.zeros(self.num_envs, device=self.device)

        # Metrics
        self.metrics["kicks_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_since_kick"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return the kick velocity command."""
        return self.kick_velocity

    def _update_metrics(self) -> None:
        """Update time since last kick."""
        self.metrics["time_since_kick"] += self._env.step_dt

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Execute a penalty kick for the specified environments.

        This samples new kick parameters and shoots the ball toward the goal.
        """
        num_envs = len(env_ids)

        # Get environment origins
        env_origins = self._env.scene.env_origins[env_ids]  # (num_envs, 3)

        # Ball starts at penalty spot (11m from goal)
        # Goal is at y=-0.5m, so penalty spot is at y=10.5m
        penalty_spot_local = torch.tensor([0.0, 10.5, 0.11], device=self.device)
        ball_start_pos = env_origins + penalty_spot_local.unsqueeze(0)  # (num_envs, 3)

        # Goal position (center of goal)
        goal_pos = self.goal.data.root_link_pos_w[env_ids]  # (num_envs, 3)

        # Sample target position within goal
        # Target X offset (left/right within goal)
        target_x_offset = torch.empty(num_envs, device=self.device).uniform_(
            *self.cfg.ranges.target_x_offset
        )

        # Target Z height (vertical position in goal)
        target_z_height = torch.empty(num_envs, device=self.device).uniform_(
            *self.cfg.ranges.target_z_height
        )

        # Target Y position (goal line to back of goal)
        target_y_offset = torch.empty(num_envs, device=self.device).uniform_(
            *self.cfg.ranges.target_y_depth
        )

        # Calculate target position
        target_pos = goal_pos.clone()
        target_pos[:, 0] += target_x_offset  # X: left/right
        target_pos[:, 1] += target_y_offset  # Y: depth into goal
        target_pos[:, 2] += target_z_height  # Z: height

        # Calculate kick velocity using ballistic trajectory
        # Distance and direction
        direction = target_pos - ball_start_pos
        horizontal_dist = torch.norm(direction[:, :2], dim=-1)  # XY distance
        vertical_dist = direction[:, 2]  # Z distance

        # Sample kick time (how long ball takes to reach goal)
        kick_time = torch.empty(num_envs, device=self.device).uniform_(
            *self.cfg.ranges.kick_time
        )

        # Ballistic trajectory calculation
        # For projectile motion with gravity:
        # v_xy = horizontal_dist / t
        # v_z = (vertical_dist + 0.5 * g * t²) / t
        gravity = 9.81  # m/s²

        kick_vel = torch.zeros(num_envs, 3, device=self.device)

        # Horizontal velocity (XY plane)
        horizontal_direction = direction[:, :2] / (horizontal_dist.unsqueeze(-1) + 1e-8)
        horizontal_speed = horizontal_dist / kick_time
        kick_vel[:, :2] = horizontal_direction * horizontal_speed.unsqueeze(-1)

        # Vertical velocity (Z axis, compensate for gravity)
        kick_vel[:, 2] = (vertical_dist + 0.5 * gravity * kick_time**2) / kick_time

        # Compute interception X position (where robot should move to)
        # This is where ball crosses the ROBOT'S goal line (y=-0.3 in local coords)
        robot_goal_line_y = -0.3  # Where robot stands
        env_origins = self._env.scene.env_origins[env_ids]
        robot_goal_line_y_world = env_origins[:, 1] + robot_goal_line_y

        # Predict where ball crosses robot's goal line
        dy_robot = robot_goal_line_y_world - ball_start_pos[:, 1]
        kick_vel_y = kick_vel[:, 1].clamp(max=-0.01)
        time_to_robot_line = dy_robot / (kick_vel_y - 1e-8)

        # Predicted X at robot's goal line (LOCAL coordinates)
        predicted_x_world = ball_start_pos[:, 0] + kick_vel[:, 0] * time_to_robot_line
        predicted_x_local = predicted_x_world - env_origins[:, 0]  # Convert to local

        # Clamp to goal width
        goal_width = 3.5
        interception_x_local = predicted_x_local.clamp(-goal_width / 2, goal_width / 2)

        # Store for visualization and for observations/rewards to use
        self.interception_x_local[env_ids] = interception_x_local
        self.kick_velocity[env_ids] = kick_vel
        self.target_position[env_ids] = target_pos
        self.ball_start_pos[env_ids] = ball_start_pos

        # Set ball position and velocity
        orientations = torch.zeros(num_envs, 4, device=self.device)
        orientations[:, 0] = 1.0  # Identity quaternion

        ang_vel = torch.zeros(num_envs, 3, device=self.device)

        # Combine into root state [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root_state = torch.cat(
            [ball_start_pos, orientations, kick_vel, ang_vel], dim=-1
        )

        # Write to simulation
        self.football.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Update metrics
        self.metrics["kicks_count"][env_ids] += 1
        self.metrics["time_since_kick"][env_ids] = 0.0

    def _update_command(self) -> None:
        """Update command state (called every step)."""
        # Nothing to update - kick parameters are set on resample
        pass

    # Visualization

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Visualize the penalty kick trajectory and interception point.

        Draws:
        - Kick velocity vector (red arrow from ball position)
        - Trajectory path to target (thin red arrow)
        - Target marker in goal (small green arrow)
        - INTERCEPTION POINT on goal line (large blue arrow) - where ball crosses goal line
        """
        batch = visualizer.env_idx

        if batch >= self.num_envs:
            return

        ball_pos = self.ball_start_pos[batch]
        kick_vel = self.kick_velocity[batch]
        target_pos = self.target_position[batch]

        # Skip if not initialized (all zeros)
        if torch.norm(kick_vel) < 1e-6:
            return

        # Draw velocity vector (thick red arrow showing kick direction and speed)
        vel_scale = 0.3  # Scale factor for visibility
        vel_end = ball_pos + kick_vel * vel_scale
        visualizer.add_arrow(ball_pos, vel_end, color=(1.0, 0.0, 0.0, 0.9), width=0.03)

        # Draw trajectory path to target (thin red arrow)
        if self.cfg.viz.show_trajectory:
            visualizer.add_arrow(
                ball_pos, target_pos, color=(1.0, 0.0, 0.0, 0.4), width=0.01
            )

        # Draw target marker in goal (small green arrow at target)
        target_marker_end = target_pos + torch.tensor(
            [0.0, 0.0, 0.2], device=self.device
        )
        visualizer.add_arrow(
            target_pos, target_marker_end, color=(0.2, 0.8, 0.2, 0.8), width=0.02
        )

        # ========================================================================
        # NEW: Draw INTERCEPTION POINT on goal line (where ball crosses Y=-0.5)
        # This is the X position the robot should move to!
        # ========================================================================
        goal_line_y = -0.5
        env_origins = self._env.scene.env_origins
        goal_line_y_world = env_origins[batch, 1] + goal_line_y

        # Predict where ball crosses goal line
        dy = goal_line_y_world - ball_pos[1]
        ball_vel_y = kick_vel[1].clamp(max=-0.01)
        time_to_goal = dy / (ball_vel_y - 1e-8)

        # Predicted X at goal line
        predicted_x = ball_pos[0] + kick_vel[0] * time_to_goal
        goal_width = 3.5
        interception_x = predicted_x.clamp(-goal_width / 2, goal_width / 2)

        # Create interception point marker on goal line
        interception_point = torch.tensor(
            [interception_x, goal_line_y_world, 0.5],  # At ground + 0.5m height
            device=self.device,
        )
        interception_marker_end = interception_point + torch.tensor(
            [0.0, 0.0, 0.5], device=self.device
        )

        # Large BLUE arrow for interception point (this is where robot should go!)
        visualizer.add_arrow(
            interception_point,
            interception_marker_end,
            color=(0.0, 0.5, 1.0, 1.0),  # Bright blue
            width=0.05,  # Extra thick to be visible
            label="Interception",
        )


@dataclass(kw_only=True)
class PenaltyKickCommandCfg(CommandTermCfg):
    """Configuration for penalty kick command term.

    This command shoots penalty kicks toward the goal from the penalty spot
    with randomized target positions and velocities.
    """

    robot_name: str = "robot"
    """Name of the robot entity."""

    football_name: str = "football_ball"
    """Name of the football entity."""

    goal_name: str = "goal"
    """Name of the goal entity."""

    @dataclass
    class Ranges:
        """Ranges for kick parameters."""

        target_x_offset: tuple[float, float] = (-3.0, 3.0)
        """X offset from goal center (meters, left/right in goal)."""

        target_z_height: tuple[float, float] = (0.3, 2.0)
        """Z height from goal bottom (meters, vertical in goal)."""

        target_y_depth: tuple[float, float] = (-0.8, -0.2)
        """Y offset from goal line (meters, depth into goal)."""

        kick_time: tuple[float, float] = (0.8, 1.5)
        """Time for ball to reach goal (seconds)."""

    ranges: Ranges = field(default_factory=Ranges)
    """Randomization ranges for kick parameters."""

    @dataclass
    class VizCfg:
        """Visualization configuration."""

        show_trajectory: bool = True
        """Whether to show the trajectory line."""

    viz: VizCfg = field(default_factory=VizCfg)
    """Visualization settings."""

    class_type: type[CommandTerm] = PenaltyKickCommand
