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
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
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
    lookat=(0.0, 0.0, 0.0),
    distance=5.0,
    elevation=-45.0,
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
            target_z_height=(
                0.3,
                1.2,
            ),  # Aim 0.3-1.2m high (reduced from 1.7 - robot can't reach that high)
            target_y_depth=(-0.8, -0.2),  # Aim into goal depth
            kick_time=(
                1.2,
                2.0,
            ),  # Slower ball: 1.2-2.0s to reach goal (easier to block)
        ),
    )


@dataclass
class ObservationCfg:
    """Simplified observations - Matching velocity task + target position."""

    @dataclass
    class PolicyCfg(ObsGroup):
        # Base observations (same as velocity task)
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

        # NEW: Target interception X position from command (where to move laterally)
        # This replaces velocity command in velocity task
        target_interception_x: ObsTerm = term(
            ObsTerm,
            func=mdp.target_interception_x_position,
            params={"command_name": "penalty_kick"},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # ±5cm noise for robustness
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
    """Simplified goalkeeper rewards - Tuned to encourage movement.

    Key insight: Robot was too comfortable standing still. Movement rewards
    must DOMINATE stability rewards to encourage lateral walking.
    """

    # POSITION is primary (be at the right place!) - Highest weight
    position_x: RewardTerm = term(
        RewardTerm,
        func=mdp.position_to_target_x,
        weight=3.0,  # PRIMARY signal - care most about being positioned correctly
        params={
            "command_name": "penalty_kick",
            "std": 0.4,
        },
    )

    # VELOCITY teaches walking - Lower weight, slower movement
    track_velocity: RewardTerm = term(
        RewardTerm,
        func=mdp.track_lateral_velocity_to_target,
        weight=1.5,  # REDUCED (was 3.0) - just helps learn locomotion
        params={
            "command_name": "penalty_kick",
            "std": 0.5,
            "max_velocity": 0.5,  # SLOWER (was 0.8) - controlled stepping
            "kp": 1.5,  # REDUCED gain (was 2.5) - gentle velocity commands
        },
    )

    # Constraints: Stay on line and standing
    stay_on_line: RewardTerm = term(
        RewardTerm,
        func=mdp.stay_on_goal_line_y,
        weight=1.0,  # Increased (was 0.5) - important constraint
        params={
            "target_y": -0.3,
            "std": 0.2,  # Tighter (was 0.3)
        },
    )

    height: RewardTerm = term(
        RewardTerm,
        func=mdp.base_height_reward,
        weight=0.5,  # Increased (was 2.0) - enforce good standing posture
        params={
            "target_height": 0.78,
            "std": 0.1,
        },
    )

    # Posture: Moderate weight for natural movement
    posture: RewardTerm = term(
        RewardTerm,
        func=mdp.posture_reward,
        weight=1.0,  # Increased (was 0.3) - enforce better movement quality
        params={
            "std": {
                # Moderate flexibility for natural walking
                r".*hip.*": 0.4,
                r".*knee.*": 0.4,
                r".*ankle.*": 0.3,
                r".*waist.*": 0.4,
                r".*shoulder.*": 0.5,
                r".*elbow.*": 0.5,
                r".*wrist.*": 0.5,
            },
        },
    )

    # Penalties: Strong smoothness constraint
    joint_limits: RewardTerm = term(
        RewardTerm,
        func=mdp.joint_limits_penalty,
        weight=-1.0,
        params={},
    )

    action_rate: RewardTerm = term(
        RewardTerm,
        func=mjlab_mdp.action_rate_l2,
        weight=-0.2,  # INCREASED (was -0.05) - strongly penalize jerky movements
        params={},
    )


@dataclass
class TerminationCfg:
    """Simplified terminations - Based on velocity task.

    Focus: Episode timeout + falling detection.
    """

    time_out: DoneTerm = term(
        DoneTerm,
        func=mjlab_mdp.time_out,
        time_out=True,
    )

    fell_over: DoneTerm = term(
        DoneTerm,
        func=mdp.fell_over,
        time_out=False,
        params={
            "limit_angle": 1.2217,  # 70 degrees (same as velocity task bad_orientation)
        },
    )

    base_too_low: DoneTerm = term(
        DoneTerm,
        func=mdp.base_too_low,
        time_out=False,
        params={
            "min_height": 0.55,  # 55cm minimum (G1 standing is ~0.78m, allow some crouch for movement)
        },
    )


@dataclass
class CurriculumCfg:
    """No curriculum for now - keep it simple like velocity task without terrain."""

    terrain_levels: CurrTerm | None = None
    command_vel: CurrTerm | None = None


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
    episode_length_s: float = 20.0  # Same as velocity task

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
        geom_names = []
        for i in range(1, 8):
            geom_names.append(f"left_foot{i}_collision")
        for i in range(1, 8):
            geom_names.append(f"right_foot{i}_collision")
        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        # Configure action scale
        self.actions.joint_pos.scale = G1_ACTION_SCALE

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
