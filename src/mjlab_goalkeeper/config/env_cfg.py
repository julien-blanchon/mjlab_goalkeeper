"""Goalkeeper robot configuration.

This module defines the complete configuration for the Unitree G1 goalkeeper robot
on a flat terrain.
"""

from dataclasses import dataclass, field, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTION_SCALE,
    G1_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg as CurrTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import ContactSensorCfg
from mjlab.viewer import ViewerConfig

import mjlab.envs.mdp as mjlab_mdp

from .. import mdp
from ..entities import FOOTBALL_BALL_ENTITY_CFG, GOAL_ENTITY_CFG


##
# Scene.
##

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
    ),
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="",  # Override in robot cfg.
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
)

##
# MDP.
##


@dataclass
class ActionCfg:
    joint_pos: mjlab_mdp.JointPositionActionCfg = term(
        mjlab_mdp.JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@dataclass
class CommandsCfg:
    penalty_kick: mdp.PenaltyKickCommandCfg = term(
        mdp.PenaltyKickCommandCfg,
        robot_name="robot",
        football_name="football_ball",
        goal_name="goal",
        resampling_time_range=(
            8.0,
            12.0,
        ),  # Wait 8-12s before each kick (learn stability first)
        debug_vis=True,
        ranges=mdp.PenaltyKickCommandCfg.Ranges(
            target_x_offset=(
                -1.5,
                1.5,
            ),  # Aim ±1.5m left/right in goal (adjusted for 3.5m width)
            target_z_height=(0.3, 1.7),  # Aim 0.3-1.7m high (adjusted for 2m height)
            target_y_depth=(-0.8, -0.2),  # Aim into goal depth
            kick_time=(
                1.2,
                2.0,
            ),  # Slower ball: 1.2-2.0s to reach goal (easier to block)
        ),
    )


@dataclass
class ObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        base_lin_vel: ObsTerm = term(
            ObsTerm,
            func=mjlab_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel: ObsTerm = term(
            ObsTerm,
            func=mjlab_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity: ObsTerm = term(
            ObsTerm,
            func=mjlab_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos: ObsTerm = term(
            ObsTerm,
            func=mjlab_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel: ObsTerm = term(
            ObsTerm,
            func=mjlab_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        actions: ObsTerm = term(ObsTerm, func=mjlab_mdp.last_action)

        # Football observations (primary ball for goalkeeper training)
        # Add small noise for robustness (simulates vision system uncertainty)
        football_position: ObsTerm = term(
            ObsTerm,
            func=mdp.entity_position,
            params={"asset_cfg": SceneEntityCfg("football_ball")},
            noise=Unoise(n_min=-0.02, n_max=0.02),  # ±2cm noise
        )
        football_velocity: ObsTerm = term(
            ObsTerm,
            func=mdp.entity_velocity,
            params={"asset_cfg": SceneEntityCfg("football_ball")},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # Small velocity noise
        )

        # Hand positions (for hand-eye coordination)
        left_hand_position: ObsTerm = term(
            ObsTerm,
            func=mdp.hand_position,
            params={"hand_name": "left_wrist_yaw_link"},
        )
        right_hand_position: ObsTerm = term(
            ObsTerm,
            func=mdp.hand_position,
            params={"hand_name": "right_wrist_yaw_link"},
        )

        # Goal detection and contact observations (1D each)
        goal_scored: ObsTerm = term(
            ObsTerm,
            func=mdp.goal_scored_detection,
        )
        robot_ball_contact: ObsTerm = term(
            ObsTerm,
            func=mdp.robot_ball_contact,
            params={"threshold": 0.5},
        )

        def __post_init__(self):
            self.enable_corruption = True

    @dataclass
    class PrivilegedCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:
    reset_base: EventTerm = term(
        EventTerm,
        func=mdp.reset_robot_on_goal_line,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_std": 0.5,  # Gaussian std for X position (centered at origin)
            "yaw_range": (
                -0.1,
                0.1,
            ),  # Very small yaw variation (±6°) - robot faces forward
        },
    )
    reset_robot_joints: EventTerm = term(
        EventTerm,
        func=mjlab_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    push_robot: EventTerm | None = term(
        EventTerm,
        func=mjlab_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(
            5.0,
            10.0,
        ),  # Push less frequently to allow learning stability
        params={
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        },  # Gentler pushes
    )
    foot_friction: EventTerm = term(
        EventTerm,
        mode="startup",
        func=mjlab_mdp.randomize_field,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", geom_names=[]
            ),  # Override in robot cfg.
            "operation": "abs",
            "field": "geom_friction",
            "ranges": (0.3, 1.2),
        },
    )


@dataclass
class RewardCfg:
    # Standard regularizers (from velocity task best practices)
    pose: RewardTerm = term(
        RewardTerm,
        func=mjlab_mdp.posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "std": [],
        },
    )
    dof_pos_limits: RewardTerm = term(
        RewardTerm, func=mjlab_mdp.joint_pos_limits, weight=-1.0
    )
    action_rate_l2: RewardTerm = term(
        RewardTerm, func=mjlab_mdp.action_rate_l2, weight=-0.1
    )
    joint_torques_l2: RewardTerm = term(
        RewardTerm,
        func=mjlab_mdp.joint_torques_l2,
        weight=-0.0001,  # Encourage energy efficiency
    )
    flat_orientation: RewardTerm = term(
        RewardTerm,
        func=mjlab_mdp.flat_orientation_l2,
        weight=-0.5,  # Penalize tilting
    )

    # Staying alive/upright reward (continuous encouragement to not fall)
    # Reduced from 1.0 to 0.4 to reduce "standing still" bias
    alive: RewardTerm = term(
        RewardTerm,
        func=mdp.is_alive,
        weight=0.4,  # Reduced to encourage movement toward ball
    )

    # New goalkeeper-specific rewards
    # Increased std from 0.3 to 1.5 for better long-range gradient (ball starts at 10.5m!)
    hand_to_ball: RewardTerm = term(
        RewardTerm,
        func=mdp.hand_to_ball_distance,
        params={
            "std": 1.5,  # Larger std gives meaningful gradient at 5-10m distances
            "left_hand_body": "left_wrist_yaw_link",
            "right_hand_body": "right_wrist_yaw_link",
        },
        weight=5.0,  # Big reward for getting hands close to ball
    )
    stabilization_after_contact: RewardTerm = term(
        RewardTerm,
        func=mdp.post_contact_stabilization,
        weight=2.0,  # Reward staying stable after blocking
        params={
            "std": 1.0,
            "contact_memory_steps": 50,  # Reward stabilization for 50 steps (1s) after contact
            "contact_threshold": 0.5,
            "nominal_height": 0.76,
        },
    )
    active_save: RewardTerm = term(
        RewardTerm,
        func=mdp.goal_prevented_by_robot,
        weight=10.0,  # Big reward for successful saves!
        params={
            "contact_threshold": 0.5,
        },
    )

    # Enabled to teach locomotion! Critical for learning to move toward ball
    air_time: RewardTerm = term(
        RewardTerm,
        func=mdp.feet_air_time,
        weight=0.5,  # Enabled! Teaches robot that lifting feet is good
        params={
            "asset_name": "robot",
            "threshold_min": 0.05,
            "threshold_max": 0.15,
            "command_name": "penalty_kick",  # Use correct command name
            "command_threshold": 0.0,  # Always enable (no velocity threshold)
            "sensor_names": [],  # Will be set in __post_init__
            "reward_mode": "continuous",  # Continuous reward while in air
        },
    )


@dataclass
class TerminationCfg:
    time_out: DoneTerm = term(DoneTerm, func=mjlab_mdp.time_out, time_out=True)

    # Re-enabled - robot must learn to stay upright
    # Combined with alive reward, this creates strong incentive to not fall
    fell_over: DoneTerm = term(
        DoneTerm,
        func=mjlab_mdp.bad_orientation,
        params={"limit_angle": 1.22},  # 70° in radians (π/180 * 70 ≈ 1.22)
    )

    goal_scored: DoneTerm = term(
        DoneTerm,
        func=mdp.goal_scored_termination,
        time_out=False,  # This is a success termination, not a timeout
    )


@dataclass
class CurriculumCfg:
    terrain_levels: CurrTerm | None = term(
        CurrTerm, func=mdp.terrain_levels_vel, params={"command_name": "twist"}
    )


##
# Environment.
##

SIM_CFG = SimulationCfg(
    nconmax=250_000,  # Sufficient for robot + football + goal
    njmax=300,
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
)


@dataclass
class UnitreeG1GoalkeeperEnvCfg(ManagerBasedRlEnvCfg):
    """Configuration for Unitree G1 robot goalkeeper environment."""

    scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
    actions: ActionCfg = field(default_factory=ActionCfg)
    rewards: RewardCfg = field(default_factory=RewardCfg)
    events: EventCfg = field(default_factory=EventCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
    viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
    decimation: int = 4  # 50 Hz control frequency.
    episode_length_s: float = 20.0

    def __post_init__(self):
        # Setup G1 robot with contact sensors for feet
        foot_contact_sensors = [
            ContactSensorCfg(
                name=f"{side}_foot_ground_contact",
                body1=f"{side}_ankle_roll_link",
                body2="terrain",
                num=1,
                data=("found",),
                reduce="netforce",
            )
            for side in ["left", "right"]
        ]

        g1_cfg = replace(G1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))

        # Add robot, football, and goal to the scene (simplified for goalkeeper training)
        self.scene.entities = {
            "robot": g1_cfg,
            "football_ball": FOOTBALL_BALL_ENTITY_CFG,
            "goal": GOAL_ENTITY_CFG,
        }

        # Configure terrain as flat plane
        assert self.scene.terrain is not None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Setup foot friction randomization
        sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
        geom_names = []
        for i in range(1, 8):
            geom_names.append(f"left_foot{i}_collision")
        for i in range(1, 8):
            geom_names.append(f"right_foot{i}_collision")
        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        # Configure action scale
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Configure rewards
        self.rewards.air_time.params["sensor_names"] = sensor_names
        self.rewards.pose.params["std"] = {
            # Lower body.
            r".*hip_pitch.*": 0.3,
            r".*hip_roll.*": 0.15,
            r".*hip_yaw.*": 0.15,
            r".*knee.*": 0.35,
            r".*ankle_pitch.*": 0.25,
            r".*ankle_roll.*": 0.1,
            # Waist.
            r".*waist_yaw.*": 0.15,
            r".*waist_roll.*": 0.08,
            r".*waist_pitch.*": 0.1,
            # Arms.
            r".*shoulder_pitch.*": 0.35,
            r".*shoulder_roll.*": 0.15,
            r".*shoulder_yaw.*": 0.1,
            r".*elbow.*": 0.25,
            r".*wrist.*": 0.3,
        }

        # Configure viewer
        self.viewer.body_name = "torso_link"
        # self.commands.twist.viz.z_offset = 0.75  # Commented out - twist command removed

        # Disable curriculum for flat terrain
        self.curriculum.terrain_levels = None
        self.curriculum.command_vel = None

        # Configure push events
        assert self.events.push_robot is not None
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }


@dataclass
class UnitreeG1GoalkeeperEnvCfg_PLAY(UnitreeG1GoalkeeperEnvCfg):
    """Configuration for Unitree G1 robot goalkeeper environment (play mode)."""

    def __post_init__(self):
        super().__post_init__()

        # Effectively infinite episode length.
        self.episode_length_s = int(1e9)
