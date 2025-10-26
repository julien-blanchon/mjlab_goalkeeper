# Goalkeeper Robot Environment

A custom mjlab environment for training a Unitree G1 humanoid robot as a goalkeeper, featuring dynamic objects (cube and ball) with smooth circular motion.

## Run with skypilot

```bash
sky launch -d -c goalkeeper-task sky_task.yaml --secret WANDB_API_KEY=<your_wandb_api_key>
```

## Features

### Entities

- **Robot**: Unitree G1 humanoid with full articulation
- **Cube**: Static red box (20cm, 0.5kg) at 2m in front of robot
- **Ball**: White sphere (22cm soccer ball, 0.45kg) moving in continuous circular motion

### Ball Circular Motion

The ball moves smoothly around the robot in a 10-meter radius circle:

- **Radius**: 10.0 meters
- **Angular velocity**: 0.3 rad/s (~17°/s)
- **Linear speed**: 3.0 m/s
- **Period**: ~21 seconds per full rotation
- **Height**: 0.11m (resting on ground)

The circular motion is maintained by periodically updating the ball's velocity (every 0.5s) to keep it tangent to the circle.

### Football Throw Command

The football is thrown at the robot using a `CommandTerm` (mjlab's proper pattern for dynamic behaviors):

- **Distance**: 8-15 meters from origin
- **Angle**: Random 360° around robot
- **Height**: 2-5 meters above ground
- **Flight time**: 1-2 seconds to reach target
- **Target**: Robot torso with ±0.5m randomization
- **Resampling**: New throw every 3-6 seconds
- **Visualization**: Orange arrows showing throw position, velocity vector, and trajectory path

### Contact Detection

**Inter-entity contacts** use **proximity-based detection** (mjlab best practice):

- `entity_contact_detected()`: Detects when entities are within threshold distance
- Works by measuring distance between entity positions
- More efficient than contact sensors for inter-entity detection
- Provides continuous distance measure (useful for rewards)

**Why not ContactSensorCfg?**

- `ContactSensorCfg` only works for:
  - Intra-entity contacts (e.g., self-collisions within robot)
  - Entity-terrain contacts
- Inter-entity contacts require programmatic detection because sensors are added to entity specs **before** scene assembly, when other entities don't exist yet.

### Custom MDP Components

#### Observations (`src/mjlab_goalkeeper/mdp/observations.py`)

- `entity_position()`: Get entity position in world frame
- `entity_velocity()`: Get entity linear velocity
- `entity_relative_position()`: Relative position between two entities
- `entity_contact_detected()`: Proximity-based contact detection

#### Events (`src/mjlab_goalkeeper/mdp/events.py`)

- `reset_ball_on_circle()`: Initialize ball on circular path (reset mode)
- `update_ball_circular_motion()`: Maintain circular motion (interval mode, every 0.5s)

## Project Structure

```
mjlab_goalkeeper/
├── src/mjlab_goalkeeper/
│   ├── __init__.py           # Gym environment registration
│   ├── entities/             # Entity configurations
│   │   ├── __init__.py
│   │   ├── cube.py          # Cube entity (static target)
│   │   └── ball.py          # Ball entity (moving target)
│   ├── mdp/                 # MDP components
│   │   ├── __init__.py
│   │   ├── events.py        # Custom events (ball motion)
│   │   └── observations.py  # Custom observations (positions, contacts)
│   └── config/
│       ├── env_cfg.py       # Environment configuration
│       └── rl_cfg.py        # RL algorithm configuration
├── train.py                 # Training script
├── play.py                  # Inference script
└── test_goalkeeper_setup.py # Validation tests
```

## Usage

### Testing the Setup

Run the comprehensive test to verify all features:

```bash
cd mjlab_goalkeeper
uv run python test_goalkeeper_setup.py
```

This tests:

- Entity creation (robot, cube, ball)
- Contact sensor configuration
- Position/velocity access
- Programmatic ball control
- Proximity-based contact detection

### Testing Circular Motion

```bash
uv run python test_circular_motion.py
```

### Testing Football Throw

```bash
uv run python test_football_throw.py
```

### Training

```bash
./train.py Mjlab-Velocity-Flat-Unitree-G1-Custom
```

### Playing/Inference

```bash
./play.py Mjlab-Velocity-Flat-Unitree-G1-Play-Custom --checkpoint_file path/to/checkpoint
```

## Accessing Entity Data in Code

### Positions and Velocities

```python
# In any MDP function (observation, reward, termination, event)
def my_custom_function(env: ManagerBasedEnv):
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    cube = env.scene["cube"]

    # Positions (num_envs, 3)
    robot_pos = robot.data.root_link_pos_w
    ball_pos = ball.data.root_link_pos_w
    cube_pos = cube.data.root_link_pos_w

    # Velocities (num_envs, 3)
    robot_vel = robot.data.root_link_lin_vel_w
    ball_vel = ball.data.root_link_lin_vel_w
    cube_vel = cube.data.root_link_lin_vel_w

    # Calculate distance
    distance = torch.norm(ball_pos - robot_pos, dim=-1)

    return distance
```

### Moving Entities Programmatically

```python
def move_ball_to_position(env, new_pos, new_vel):
    ball = env.scene["ball"]

    # Create root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    new_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    new_ang_vel = torch.zeros(1, 3, device=env.device)

    root_state = torch.cat([
        new_pos,      # (1, 3)
        new_quat,     # (1, 4)
        new_vel,      # (1, 3)
        new_ang_vel   # (1, 3)
    ], dim=-1)

    # Write to environment 0
    ball.write_root_state_to_sim(root_state, env_ids=torch.tensor([0]))

    # Must call forward() to update derived quantities
    env.sim.forward()
```

### Contact Detection

```python
from mjlab_goalkeeper.mdp import entity_contact_detected
from mjlab.managers.scene_entity_config import SceneEntityCfg

def check_contacts(env: ManagerBasedEnv):
    # Proximity-based detection (default threshold: 0.3m)
    robot_ball_contact = entity_contact_detected(
        env,
        SceneEntityCfg("robot"),
        SceneEntityCfg("ball"),
        threshold=0.5  # Custom threshold
    )
    # Returns (num_envs,) tensor: 1.0 if contact, 0.0 otherwise

    # Use in reward function
    contact_reward = robot_ball_contact * 10.0  # Reward for touching ball

    return contact_reward
```

## Technical Details

### Why Proximity-Based Contact Detection?

1. **ContactSensorCfg Limitation**: Contact sensors are added to entity specs during initialization, before scene assembly. At that point, other entities (cube, ball) don't exist in the robot's spec yet.

2. **Terrain Exception**: Only `"terrain"` works as `body2` because it's added directly to the scene, not as a prefixed entity.

3. **Programmatic Approach**: For inter-entity contacts, mjlab best practice is:
   - Access entity positions via `entity.data.root_link_pos_w`
   - Calculate distance between entities
   - Apply threshold for contact detection
   - More flexible and provides continuous distance measure

### Circular Motion Implementation

The ball's circular motion uses two complementary events:

1. **Reset Event** (`reset_ball_on_circle`):

   - Places ball at starting position (10m, 0°)
   - Sets initial tangential velocity
   - Called on episode reset

2. **Interval Event** (`update_ball_circular_motion`):
   - Updates ball velocity every 0.5 seconds
   - Recalculates tangent direction based on current position
   - Corrects for drift due to physics interactions
   - Maintains smooth circular trajectory

The velocity is continuously updated rather than using forces because:

- More predictable and stable
- No accumulation of energy errors
- Easier to tune (directly set speed, not force)
- Works reliably with varying simulation parameters

## Customization

### Adjusting Ball Speed

Edit `src/mjlab_goalkeeper/config/env_cfg.py`:

```python
reset_ball: EventTerm = term(
    EventTerm,
    func=mdp.reset_ball_on_circle,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("ball"),
        "radius": 10.0,
        "angular_velocity": 0.5,  # Faster: 0.5 rad/s
        "height": 0.11,
    },
)
update_ball_motion: EventTerm = term(
    EventTerm,
    func=mdp.update_ball_circular_motion,
    mode="interval",
    interval_range_s=(0.5, 0.5),
    params={
        "asset_cfg": SceneEntityCfg("ball"),
        "radius": 10.0,
        "angular_velocity": 0.5,  # Match reset value
        "height": 0.11,
    },
)
```

### Adding Ball Observations to Policy

Edit `ObservationCfg.PolicyCfg` in `env_cfg.py`:

```python
ball_relative_pos: ObsTerm = term(
    ObsTerm,
    func=mdp.entity_relative_position,
    params={
        "target_asset_cfg": SceneEntityCfg("ball"),
        "reference_asset_cfg": SceneEntityCfg("robot"),
    },
)
```

### Adding Contact Rewards

Edit `RewardCfg` in `env_cfg.py`:

```python
ball_contact: RewardTerm = term(
    RewardTerm,
    func=lambda env: mdp.entity_contact_detected(
        env,
        SceneEntityCfg("robot"),
        SceneEntityCfg("ball"),
        threshold=0.5
    ),
    weight=5.0,
)
```

## Testing & Validation

All tests validated ✅:

- Entity initialization at correct positions
- Circular motion maintains 10m radius (std dev: 0.043m)
- Programmatic position control
- Proximity-based contact detection
- Observation access to positions and velocities

Run tests anytime to validate changes:

```bash
uv run python test_goalkeeper_setup.py
uv run python test_circular_motion.py
```
