# MJLab Developer Guide: Goalkeeper Project

This guide explains how the mjlab framework works using the goalkeeper project as a concrete example. Perfect for developers onboarding to mjlab or creating custom tasks.

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Project Architecture](#project-architecture)
3. [Creating Custom Entities](#creating-custom-entities)
4. [MDP Components](#mdp-components)
5. [Environment Configuration](#environment-configuration)
6. [Training & Inference](#training--inference)
7. [Testing Strategy](#testing-strategy)
8. [Best Practices](#best-practices)

---

## Framework Overview

### What is mjlab?

mjlab is a **manager-based RL framework** for robotics built on MuJoCo and GPU-accelerated simulation (mujoco-warp). It uses a declarative configuration system where you define:

- **Entities**: Physical objects (robots, props, obstacles)
- **MDP Components**: Observations, actions, rewards, terminations, commands, events
- **Scene**: How entities are arranged and configured
- **Simulation**: Physics parameters and contact settings

### Key Philosophy

1. **Manager-Based**: Each MDP component (observations, rewards, etc.) is managed by a dedicated manager that orchestrates multiple "terms"
2. **Declarative Configuration**: Use dataclasses and the `term()` helper instead of imperative code
3. **Modular**: MDP terms are reusable functions that work across different tasks
4. **GPU-First**: Designed for massive parallelization (thousands of environments)

---

## Project Architecture

### Directory Structure

```
mjlab_goalkeeper/
├── src/mjlab_goalkeeper/
│   ├── __init__.py              # Gym registration
│   ├── entities/                # Entity definitions
│   │   ├── __init__.py
│   │   ├── cube.py             # Static target
│   │   ├── ball.py             # Circular motion ball
│   │   └── football_ball.py    # Thrown ball
│   ├── mdp/                    # MDP components
│   │   ├── __init__.py         # Exports all MDP functions
│   │   ├── commands.py         # Custom commands (football throw)
│   │   ├── events.py           # Custom events (ball motion)
│   │   ├── observations.py     # Custom observations (entity positions)
│   │   ├── rewards.py          # Custom rewards (velocity tracking)
│   │   ├── curriculums.py      # Curriculum learning
│   │   └── terminations.py     # Termination conditions
│   ├── rl/                     # RL-specific code
│   │   ├── exporter.py         # ONNX export
│   │   └── runner.py           # Training runner
│   └── config/                 # Configuration
│       ├── env_cfg.py          # Environment config (MDP + Scene)
│       └── rl_cfg.py           # RL algorithm config (PPO)
├── train.py                    # Training entry point
├── play.py                     # Inference entry point
└── test_*.py                   # Validation tests
```

### Import Hierarchy

```
mjlab (base framework)
  ├── mjlab.envs.mdp (base MDP functions)
  └── mjlab.tasks.velocity.mdp (velocity tracking MDP)
      └── mjlab_goalkeeper.mdp (goalkeeper-specific MDP)
          ├── Inherits all base + velocity functions
          └── Adds custom: FootballThrowCommand, entity observations, ball events
```

**Key Pattern**: `src/mjlab_goalkeeper/mdp/__init__.py` imports everything:

```python
from mjlab.envs.mdp import *  # Base MDP
from mjlab.tasks.velocity.mdp import *  # Velocity-specific
from mjlab_goalkeeper.mdp.commands import FootballThrowCommand  # Custom
```

This makes all MDP functions available as `mdp.function_name()` in your config.

---

## Creating Custom Entities

### Entity Types (Following MuJoCo Physics)

mjlab supports 4 entity types:

| Type                     | Example            | Fixed Base | Articulated | Actuated |
| ------------------------ | ------------------ | ---------- | ----------- | -------- |
| Fixed Non-articulated    | Table, wall        | ✓          | ✗           | ✗        |
| Fixed Articulated        | Robot arm          | ✓          | ✓           | ✓/✗      |
| Floating Non-articulated | **Ball, cube**     | ✗          | ✗           | ✗        |
| Floating Articulated     | **Humanoid robot** | ✗          | ✓           | ✓/✗      |

### Example: Creating a Ball Entity

**File**: `src/mjlab_goalkeeper/entities/ball.py`

```python
import mujoco
from mjlab.entity import EntityCfg

def create_ball_spec() -> mujoco.MjSpec:
    """Create a ball (sphere) spec."""
    spec = mujoco.MjSpec()

    # Create body with freejoint (enables 6-DOF movement)
    body = spec.worldbody.add_body(name="ball_body")
    body.add_freejoint(name="ball_freejoint")

    # Add sphere geometry
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.11, 0.11, 0.11],  # 22cm diameter
        rgba=[1.0, 1.0, 1.0, 1.0],  # White
        mass=0.45,  # 450g
        friction=[0.8, 0.005, 0.0001],
    )

    return spec

BALL_ENTITY_CFG = EntityCfg(
    spec_fn=create_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(10.0, 0.0, 0.11),  # Initial position
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quat
        lin_vel=(0.0, 3.0, 0.0),  # Initial velocity
    ),
)
```

**Key Points**:

- `spec_fn`: Returns a `mujoco.MjSpec` (can be from XML file or programmatic)
- Floating base → needs `add_freejoint()`
- `init_state`: Default state (used during reset)

### Accessing Entity Data

```python
ball = env.scene["ball"]

# Position & velocity (num_envs, 3)
pos = ball.data.root_link_pos_w
vel = ball.data.root_link_lin_vel_w

# For articulated entities
joint_pos = robot.data.joint_pos  # (num_envs, num_joints)
joint_vel = robot.data.joint_vel

# Body-level data (num_envs, num_bodies, 3)
body_positions = robot.data.body_link_pos_w
```

### Controlling Entity State

```python
# Write position/velocity
new_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)  # (num_envs, 13)
ball.write_root_state_to_sim(new_state, env_ids=env_ids)

# Must call forward() to update derived quantities
env.sim.forward()
```

---

## MDP Components

mjlab uses **Manager-based MDP** where each component type has a dedicated manager.

### 1. Observations

**Purpose**: Define what the policy observes

**File**: `src/mjlab_goalkeeper/mdp/observations.py`

```python
def entity_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get entity position in world frame."""
    entity = env.scene[asset_cfg.name]
    return entity.data.root_link_pos_w  # (num_envs, 3)
```

**Configuration**: `config/env_cfg.py`

```python
@dataclass
class ObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        ball_position: ObsTerm = term(
            ObsTerm,
            func=mdp.entity_position,
            params={"asset_cfg": SceneEntityCfg("ball")},
            noise=Unoise(n_min=-0.1, n_max=0.1),  # Optional noise
        )
```

**Observation Groups**:

- `policy`: What the actor network sees (may have noise)
- `critic`: What the critic sees (usually privileged, no noise)

### 2. Actions

**Purpose**: How the policy controls the robot

**Configuration**: `config/env_cfg.py`

```python
@dataclass
class ActionCfg:
    joint_pos: mdp.JointPositionActionCfg = term(
        mdp.JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],  # Regex pattern
        scale=0.5,  # Action space scaling
        use_default_offset=True,  # Offset by default joint positions
    )
```

Actions are typically predefined in mjlab (joint position, velocity, torque).

### 3. Commands

**Purpose**: Generate targets for the robot to track (velocity, position, etc.)

Commands are **resampled periodically** to provide varying tasks.

**Example: Football Throw Command**

**File**: `src/mjlab_goalkeeper/mdp/commands.py`

```python
class FootballThrowCommand(CommandTerm):
    """Periodically throws a football at the robot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.robot_name]
        self.football = env.scene[cfg.football_name]
        # Initialize metrics
        self.metrics["throws_count"] = torch.zeros(...)

    @property
    def command(self) -> torch.Tensor:
        """Return current command (for observation functions)."""
        return self.throw_velocity

    def _resample_command(self, env_ids: torch.Tensor):
        """Called periodically to generate new throw."""
        # Sample throw position, velocity
        # Reset football position/velocity
        self.football.write_root_state_to_sim(...)

    def _update_command(self):
        """Called every step to update command state."""
        pass

    def _debug_vis_impl(self, visualizer):
        """Visualize command using arrows."""
        visualizer.add_arrow(start, end, color, width)
```

**Configuration**:

```python
@dataclass
class CommandsCfg:
    football_throw: mdp.FootballThrowCommandCfg = term(
        mdp.FootballThrowCommandCfg,
        robot_name="robot",
        football_name="football_ball",
        resampling_time_range=(3.0, 6.0),  # Resample every 3-6s
        debug_vis=True,  # Enable visualization
        ranges=mdp.FootballThrowCommandCfg.Ranges(
            distance=(8.0, 15.0),
            angle=(-3.14, 3.14),
            # ... other ranges
        ),
    )
```

**Visualization**: Only `add_arrow()` and `add_ghost_mesh()` are supported by `DebugVisualizer` interface.

### 4. Events

**Purpose**: State modifications that happen at specific times

**Three modes**:

- `startup`: Once at initialization
- `reset`: Every episode reset
- `interval`: Periodically during episodes

**Example: Ball Circular Motion**

**File**: `src/mjlab_goalkeeper/mdp/events.py`

```python
def update_ball_circular_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    radius: float = 10.0,
    angular_velocity: float = 0.3,
) -> None:
    """Update ball velocity to maintain circular motion."""
    ball = env.scene[asset_cfg.name]

    # Calculate tangent velocity based on current position
    # ... (see events.py for implementation)

    ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
```

**Configuration**:

```python
@dataclass
class EventCfg:
    reset_ball: EventTerm = term(
        EventTerm,
        func=mdp.reset_ball_on_circle,
        mode="reset",  # Called on episode reset
        params={"asset_cfg": SceneEntityCfg("ball"), "radius": 10.0},
    )

    update_ball_motion: EventTerm = term(
        EventTerm,
        func=mdp.update_ball_circular_motion,
        mode="interval",  # Called periodically
        interval_range_s=(0.5, 0.5),  # Every 0.5s
        params={"asset_cfg": SceneEntityCfg("ball"), "radius": 10.0},
    )
```

### 5. Rewards

**Purpose**: Provide learning signal to the policy

**File**: `src/mjlab_goalkeeper/mdp/rewards.py`

```python
def track_lin_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking linear velocity command."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    actual = asset.data.root_link_lin_vel_b
    desired = torch.zeros_like(actual)
    desired[:, :2] = command[:, :2]

    error = torch.sum(torch.square(desired - actual), dim=1)
    return torch.exp(-error / std**2)  # (num_envs,)
```

**Configuration**:

```python
@dataclass
class RewardCfg:
    track_lin_vel: RewardTerm = term(
        RewardTerm,
        func=mdp.track_lin_vel_exp,
        weight=1.0,  # Reward coefficient
        params={"command_name": "twist", "std": 0.5},
    )
```

### 6. Terminations

**Purpose**: Decide when episodes end

**Configuration**:

```python
@dataclass
class TerminationCfg:
    time_out: DoneTerm = term(
        DoneTerm,
        func=mdp.time_out,
        time_out=True  # Marks as timeout (not failure)
    )
    fell_over: DoneTerm = term(
        DoneTerm,
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)}
    )
```

### 7. Curriculum

**Purpose**: Gradually increase task difficulty

**File**: `src/mjlab_goalkeeper/mdp/curriculums.py`

```python
def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    """Move successful robots to harder terrains."""
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - ..., dim=1)
    move_up = distance > threshold
    terrain.update_env_origins(env_ids, move_up, move_down)
```

---

## Environment Configuration

### Main Configuration File: `config/env_cfg.py`

This is the **heart** of your task. It combines all MDP components into a cohesive environment.

#### Structure

```python
##
# Scene Setup
##

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="torso_link",
)

##
# MDP Components (as dataclasses)
##

@dataclass
class ActionCfg:
    joint_pos: mdp.JointPositionActionCfg = term(...)

@dataclass
class CommandsCfg:
    twist: mdp.UniformVelocityCommandCfg = term(...)
    football_throw: mdp.FootballThrowCommandCfg = term(...)

@dataclass
class ObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        base_lin_vel: ObsTerm = term(...)
        ball_position: ObsTerm = term(...)
        # ... more observations

    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: PolicyCfg = field(default_factory=PolicyCfg)

@dataclass
class EventCfg:
    reset_ball: EventTerm = term(...)
    update_ball_motion: EventTerm = term(...)

@dataclass
class RewardCfg:
    track_vel: RewardTerm = term(...)

@dataclass
class TerminationCfg:
    time_out: DoneTerm = term(...)

##
# Environment Class
##

@dataclass
class UnitreeG1GoalkeeperEnvCfg(ManagerBasedRlEnvCfg):
    """Main environment configuration."""

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
    decimation: int = 4  # 50 Hz control (200 Hz physics / 4)
    episode_length_s: float = 20.0

    def __post_init__(self):
        """Runtime configuration adjustments."""
        # Add entities to scene
        self.scene.entities = {
            "robot": g1_cfg,
            "cube": CUBE_ENTITY_CFG,
            "ball": BALL_ENTITY_CFG,
            "football_ball": FOOTBALL_BALL_ENTITY_CFG,
        }

        # Adjust parameters dynamically
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.viewer.body_name = "torso_link"
```

#### The `term()` Helper

`term()` is syntactic sugar for creating config objects:

```python
# Without term()
ObsTerm(
    func=mdp.base_lin_vel,
    noise=Unoise(n_min=-0.1, n_max=0.1)
)

# With term() - cleaner!
term(
    ObsTerm,
    func=mdp.base_lin_vel,
    noise=Unoise(n_min=-0.1, n_max=0.1)
)
```

### Simulation Configuration

```python
SIM_CFG = SimulationCfg(
    nconmax=250_000,  # Max contacts (increase with more entities!)
    njmax=300,  # Max constraints
    mujoco=MujocoCfg(
        timestep=0.005,  # 200 Hz physics
        iterations=10,  # Solver iterations
        ls_iterations=20,  # Line search iterations
    ),
)
```

**Critical**: `nconmax` must be large enough for `num_envs * contacts_per_env`. With 4096 envs and complex robots, you need 250k+.

---

## MDP Component Deep Dive

### Commands vs Events: When to Use Each?

| Feature           | Commands                           | Events                         |
| ----------------- | ---------------------------------- | ------------------------------ |
| **Purpose**       | Generate goals/targets             | Modify environment state       |
| **Resampling**    | Periodic (controlled by user)      | Startup/Reset/Interval         |
| **Visualization** | Yes (arrows, ghosts)               | No                             |
| **Metrics**       | Track tracking error               | No tracking                    |
| **Example**       | Velocity commands, motion commands | Push robot, randomize friction |

**Use Commands** when:

- You want the robot to track/achieve something
- You need visualization of the target
- You want metrics on tracking performance

**Use Events** when:

- You're modifying physics (friction, forces)
- You're resetting object states
- You don't need visualization

**Example: Football Ball**

We use a **Command** (not an Event) because:

- ✓ We want to visualize the throw trajectory
- ✓ We could track if the robot blocks it (future reward)
- ✓ It's a "task" the robot responds to
- ✓ Commands feel more natural for dynamic behaviors the robot observes

### Writing Custom Observations

**Pattern**:

```python
def my_observation(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    some_parameter: float = 1.0,
) -> torch.Tensor:
    """Documentation.

    Returns:
        torch.Tensor of shape (num_envs, obs_dim)
    """
    entity = env.scene[asset_cfg.name]
    # Compute observation
    obs = ...  # Some tensor operation
    return obs  # (num_envs, obs_dim)
```

**Key Requirements**:

- First arg: `env: ManagerBasedEnv`
- Optional args: Go in `params={}` dict
- Return: `torch.Tensor` with shape `(num_envs, ...)`

### Writing Custom Rewards

**Pattern**:

```python
def my_reward(
    env: ManagerBasedEnv,
    weight_param: float = 1.0,
) -> torch.Tensor:
    """Compute reward signal.

    Returns:
        torch.Tensor of shape (num_envs,)
    """
    # Compute reward (scalar per environment)
    reward = ...
    return reward  # (num_envs,)
```

**Stateful Rewards**: Use a class instead of a function

```python
class feet_air_time:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        # Initialize state
        self.current_air_time = torch.zeros(env.num_envs, num_feet)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        # Update state and compute reward
        self.current_air_time += env.step_dt
        return reward

    def reset(self, env_ids):
        # Reset state for terminated environments
        self.current_air_time[env_ids] = 0.0
```

### Inter-Entity Contact Detection

**Important**: `ContactSensorCfg` only works for:

- Intra-entity contacts (robot self-collision)
- Entity-terrain contacts

For **inter-entity contacts** (robot-ball), use programmatic detection:

```python
def entity_contact_detected(
    env: ManagerBasedEnv,
    entity1_cfg: SceneEntityCfg,
    entity2_cfg: SceneEntityCfg,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Proximity-based contact detection."""
    pos1 = env.scene[entity1_cfg.name].data.root_link_pos_w
    pos2 = env.scene[entity2_cfg.name].data.root_link_pos_w
    distance = torch.norm(pos2 - pos1, dim=-1)
    return (distance < threshold).float()
```

**Why?**: Sensors are added to entity specs **before** scene assembly. At that point, other entities don't exist yet.

---

## Training & Inference

### Gym Registration

**File**: `src/mjlab_goalkeeper/__init__.py`

```python
import gymnasium as gym

gym.register(
    id="Mjlab-Velocity-Flat-Unitree-G1-Custom",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.env_cfg:UnitreeG1GoalkeeperEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.config.rl_cfg:UnitreeG1PPORunnerCfg",
    },
)
```

### Training Entry Point

**File**: `train.py`

```python
import mjlab_goalkeeper  # Registers environments

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
```

**Usage**:

```bash
# Train with default settings
./train.py Mjlab-Velocity-Flat-Unitree-G1-Custom

# Override config via CLI
./train.py Mjlab-Velocity-Flat-Unitree-G1-Custom \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 50000 \
    --device cuda:0
```

### Inference/Play

**File**: `play.py`

```python
import mjlab_goalkeeper

from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
```

**Usage**:

```bash
./play.py Mjlab-Velocity-Flat-Unitree-G1-Play-Custom \
    --checkpoint-file logs/rsl_rl/g1_velocity/model_1000.pt \
    --viewer viser
```

---

## Testing Strategy

### Test Pyramid

```
           ┌─────────────────────┐
           │  Integration Tests  │  ← Full env + training
           │  (slow, rare)       │
           └─────────────────────┘
        ┌─────────────────────────────┐
        │    Component Tests          │  ← MDP components
        │    (medium speed)            │
        └─────────────────────────────┘
    ┌───────────────────────────────────────┐
    │     Unit Tests                        │  ← Individual functions
    │     (fast, frequent)                  │
    └───────────────────────────────────────┘
```

### Our Test Files

#### 1. `test_goalkeeper_setup.py` - Integration Test

**Purpose**: Validate complete environment setup

**Tests**:

- ✓ All entities created correctly
- ✓ Contact sensors configured
- ✓ Observations accessible
- ✓ Programmatic control works
- ✓ Contact detection works

**Run**: `uv run python test_goalkeeper_setup.py`

**Speed**: ~10-15 seconds (compiles Warp kernels)

#### 2. `test_circular_motion.py` - Component Test

**Purpose**: Validate ball circular motion

**Tests**:

- ✓ Ball initializes at correct radius
- ✓ Ball maintains circular path (low variance)
- ✓ Velocity magnitude correct
- ✓ Programmatic control works

**Run**: `uv run python test_circular_motion.py`

#### 3. `test_football_throw.py` - Component Test

**Purpose**: Validate football throw command

**Tests**:

- ✓ Football entity created
- ✓ Command registered
- ✓ Throw physics correct
- ✓ Ballistic trajectory works
- ✓ Command resampling works

**Run**: `uv run python test_football_throw.py`

### Testing Best Practices

#### 1. Use CUDA for Speed

```python
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")  # Fast!
# vs
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")  # Slow
```

CUDA is **10-100x faster** even for single environment due to Warp compilation.

#### 2. Test Incrementally

When adding features:

```bash
# 1. Add entity → test entity creation
uv run python test_goalkeeper_setup.py

# 2. Add event → test motion
uv run python test_circular_motion.py

# 3. Add command → test command
uv run python test_football_throw.py

# 4. Full integration → test training
./train.py MyTask --env.scene.num-envs 16 --agent.max-iterations 10
```

#### 3. Test Structure Pattern

```python
def test_feature():
    print("=" * 70)
    print("Testing Feature X")
    print("=" * 70)

    # Setup
    env = create_env()

    # Test 1: Basic functionality
    print("\n[1/N] Test description...")
    result = test_something()
    assert condition, "Error message"
    print("✓ Test passed")

    # Test 2: Edge cases
    print("\n[2/N] Test edge case...")
    # ...

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
```

#### 4. Debugging Failed Tests

```python
# Print tensor shapes
print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

# Check for NaNs
assert not torch.isnan(tensor).any(), f"NaN detected in {tensor}"

# Inspect entity state
ball = env.scene["ball"]
print(f"Position: {ball.data.root_link_pos_w[0]}")
print(f"Velocity: {ball.data.root_link_lin_vel_w[0]}")

# Check command manager
print(f"Active commands: {list(env.command_manager._terms.keys())}")
cmd = env.command_manager.get_command("twist")
print(f"Command value: {cmd[0]}")
```

---

## Best Practices

### 1. Entity Organization

**DO**: Separate entity files

```
entities/
  ├── cube.py
  ├── ball.py
  └── football_ball.py
```

**DON'T**: Define entities inline in `env_cfg.py`

### 2. MDP Function Naming

Follow mjlab conventions:

- Observations: `entity_position()`, `base_lin_vel()`
- Rewards: `track_lin_vel_exp()`, `foot_clearance_reward()`
- Events: `reset_ball_on_circle()`, `push_robot()`
- Commands: `UniformVelocityCommand`, `FootballThrowCommand`

### 3. Import Structure

**File**: `mdp/__init__.py`

```python
# Layer imports (broad to specific)
from mjlab.envs.mdp import *  # Base
from mjlab.tasks.velocity.mdp import *  # Task-specific
from mjlab_goalkeeper.mdp.commands import *  # Custom

# Explicit exports
__all__ = [
    "FootballThrowCommand",
    "entity_position",
    # ...
]
```

This makes everything available as `mdp.function_name()` in configs.

### 4. Configuration Tuning

**Start with small scale**:

```bash
# Development/testing: 16 envs
./train.py MyTask --env.scene.num-envs 16 --agent.max-iterations 100

# Full training: 4096 envs
./train.py MyTask --env.scene.num-envs 4096 --agent.max-iterations 30000
```

### 5. nconmax Estimation

Rule of thumb:

```python
nconmax = num_envs * contacts_per_env * safety_factor

# Example:
# - 4096 envs
# - ~40 contacts per env (robot + objects)
# - 1.5x safety factor
nconmax = 4096 * 40 * 1.5 = 245,760 ≈ 250,000
```

Increase if you see: `ValueError: nconmax overflow`

---

## Common Patterns

### Adding a New Observation

1. **Write function** in `mdp/observations.py`:

```python
def my_obs(env: ManagerBasedEnv, param: float = 1.0) -> torch.Tensor:
    return some_tensor  # (num_envs, obs_dim)
```

2. **Export** in `mdp/__init__.py`:

```python
from mjlab_goalkeeper.mdp.observations import my_obs
__all__ = [..., "my_obs"]
```

3. **Configure** in `config/env_cfg.py`:

```python
my_observation: ObsTerm = term(
    ObsTerm,
    func=mdp.my_obs,
    params={"param": 2.0},
)
```

4. **Test**:

```python
obs = env.reset()
assert "my_observation" in obs["policy"]  # Check it exists
print(obs["policy"]["my_observation"].shape)  # Check shape
```

### Adding a New Event

1. **Write function** in `mdp/events.py`
2. **Export** in `mdp/__init__.py`
3. **Configure** in `EventCfg` with appropriate mode
4. **Test**: Create env, reset, check entity states changed

### Adding a New Entity

1. **Create** `entities/my_entity.py`
2. **Export** in `entities/__init__.py`
3. **Import** in `config/env_cfg.py`
4. **Add to scene** in `__post_init__()`:

```python
self.scene.entities = {
    "robot": ...,
    "my_entity": MY_ENTITY_CFG,
}
```

5. **Test**: Check `env.scene.entities` contains it

---

## Troubleshooting

### Issue: `AttributeError: module 'mdp' has no attribute 'X'`

**Cause**: Function not exported in `mdp/__init__.py`

**Fix**:

```python
from mjlab_goalkeeper.mdp.my_module import my_function
__all__ = [..., "my_function"]
```

### Issue: `nconmax overflow`

**Cause**: Too many contacts for allocated buffer

**Fix**: Increase `nconmax` in `SIM_CFG`:

```python
SIM_CFG = SimulationCfg(
    nconmax=300_000,  # Increase this
)
```

### Issue: `KeyError: 'entity_name'` in scene

**Cause**: Entity not added to `self.scene.entities`

**Fix**: Add in `__post_init__()`:

```python
self.scene.entities = {
    "robot": robot_cfg,
    "entity_name": ENTITY_CFG,  # Add this
}
```

### Issue: Entity at (0,0,0) instead of configured position

**Cause**: No reset event for that entity

**Fix**: Add reset event:

```python
reset_my_entity: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {},  # No randomization
        "asset_cfg": SceneEntityCfg("my_entity"),
    },
)
```

### Issue: Visualization error with `add_sphere()`, `add_line()`

**Cause**: `DebugVisualizer` only supports `add_arrow()` and `add_ghost_mesh()`

**Fix**: Use arrows creatively:

```python
# Instead of sphere: use very short arrow
visualizer.add_arrow(pos, pos + [0, 0, 0.01], color, width=0.05)

# Instead of line: use thin arrow
visualizer.add_arrow(start, end, color, width=0.005)

# Marker: small upward arrow
visualizer.add_arrow(pos, pos + [0, 0, 0.3], color, width=0.02)
```

---

## Development Workflow

### 1. Create New Task

```bash
# 1. Define entities
mkdir -p src/my_task/entities
# Create entity files

# 2. Define MDP components
mkdir -p src/my_task/mdp
# Create observations.py, rewards.py, etc.

# 3. Create configuration
mkdir -p src/my_task/config
# Create env_cfg.py, rl_cfg.py

# 4. Register task
# Edit src/my_task/__init__.py

# 5. Create test
# Create test_my_task.py
```

### 2. Iterative Development

```bash
# Test cycle (fast iteration)
edit code → uv run python test_my_feature.py → fix → repeat

# Integration test
uv run python test_my_task_setup.py

# Training test (small scale)
./train.py MyTask --env.scene.num-envs 16 --agent.max-iterations 50

# Full training
./train.py MyTask --env.scene.num-envs 4096
```

### 3. Visualization During Development

```bash
# Play with random policy (test scene/entities)
./play.py MyTask-Play --num-episodes 1

# Play with trained policy
./play.py MyTask-Play --checkpoint-file path/to/model.pt
```

---

## Key Takeaways

1. **Entities** = Physical objects defined via `EntityCfg` + `spec_fn`
2. **MDP Components** = Functions that take `env` and return tensors
3. **Configuration** = Dataclasses that wire everything together
4. **Commands** = Managed goals/targets with visualization
5. **Events** = State modifications at specific times
6. **Testing** = Start small (entities) → build up (components) → integrate (full env)
7. **Use CUDA** for all tests (even single env is 10-100x faster)
8. **Contact Detection** = Proximity-based for inter-entity, sensors for intra-entity/terrain

---

## Quick Reference

### Accessing Environment Data

```python
# Scene
env.scene["entity_name"]  # Get entity
env.scene.env_origins  # Environment positions
env.scene.entities  # Dict of all entities

# Managers
env.command_manager.get_command("command_name")
env.observation_manager.compute()
env.reward_manager.compute()

# Simulation
env.sim.data  # WarpBridge to mjwarp.Data
env.sim.model  # WarpBridge to mjwarp.Model
env.sim.mj_model  # Original mujoco.MjModel
env.sim.forward()  # Update derived quantities
```

### Common Operations

```python
# Move entity
entity.write_root_state_to_sim(state, env_ids)
env.sim.forward()  # Important!

# Sample random values
value = torch.empty(num_envs, device=env.device).uniform_(min, max)

# Get entity distance
distance = torch.norm(entity1.data.root_link_pos_w - entity2.data.root_link_pos_w, dim=-1)

# Check if close (contact)
contact = (distance < threshold).float()
```

---

## Example: Adding a New Feature

Let's add a "ball catch reward" that rewards the robot for being close to the thrown football.

### Step 1: Write Reward Function

**File**: `src/mjlab_goalkeeper/mdp/rewards.py`

```python
def football_catch_proximity(
    env: ManagerBasedEnv,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for being close to the thrown football."""
    robot = env.scene["robot"]
    football = env.scene["football_ball"]

    distance = torch.norm(
        football.data.root_link_pos_w - robot.data.root_link_pos_w,
        dim=-1
    )

    # Exponential reward (peaks when distance = 0)
    return torch.exp(-distance / std**2)
```

### Step 2: Export Function

**File**: `src/mjlab_goalkeeper/mdp/__init__.py`

```python
from mjlab_goalkeeper.mdp.rewards import football_catch_proximity

__all__ = [..., "football_catch_proximity"]
```

### Step 3: Configure Reward

**File**: `src/mjlab_goalkeeper/config/env_cfg.py`

```python
@dataclass
class RewardCfg:
    # ... existing rewards ...

    football_catch: RewardTerm = term(
        RewardTerm,
        func=mdp.football_catch_proximity,
        weight=2.0,  # Reward coefficient
        params={"std": 0.5},
    )
```

### Step 4: Test

```python
env = create_env()
obs, info = env.reset()
obs, reward, _, _, _ = env.step(action)

print(f"Reward: {reward[0]}")  # Should be non-zero when close to football
```

Done! The reward is now active during training.

---

## Additional Resources

- **mjlab documentation**: https://github.com/mujocolab/mjlab
- **MuJoCo documentation**: https://mujoco.readthedocs.io/
- **mujoco-warp**: GPU acceleration layer
- **Example tasks**: `mjlab/src/mjlab/tasks/` (velocity, tracking)

---

## Summary

**mjlab in 5 steps**:

1. **Define entities** (`entities/*.py`) - Physical objects
2. **Write MDP functions** (`mdp/*.py`) - Observations, rewards, events, commands
3. **Configure environment** (`config/env_cfg.py`) - Wire it all together
4. **Register with Gym** (`__init__.py`) - Make it discoverable
5. **Train** (`./train.py MyTask`) - Learn the policy!

The goalkeeper project demonstrates all these concepts with:

- ✅ 4 entities (robot, cube, ball, football)
- ✅ 2 commands (velocity, football throw)
- ✅ Multiple events (reset, circular motion)
- ✅ Custom observations (entity positions/velocities)
- ✅ Contact detection (proximity-based)
- ✅ Comprehensive tests

Happy developing! 🎯⚽🤖
