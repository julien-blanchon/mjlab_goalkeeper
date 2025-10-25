"""Football ball entity for thrown ball goalkeeper training."""

import mujoco

from mjlab.entity import EntityCfg


def create_football_ball_spec() -> mujoco.MjSpec:
    """Create a football (soccer ball) spec for throwing at the robot.

    This ball will be periodically thrown at the robot from various positions
    to train goalkeeper blocking and catching behaviors.

    Returns:
        MjSpec: MuJoCo specification for the football.
    """
    spec = mujoco.MjSpec()

    # Create a body with a freejoint (enables 6-DOF movement)
    body = spec.worldbody.add_body(name="football_body")
    body.add_freejoint(name="football_freejoint")

    # Add a sphere geometry (soccer ball)
    body.add_geom(
        name="football_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.11, 0.11, 0.11],  # 22cm diameter (standard soccer ball)
        rgba=[0.9, 0.5, 0.1, 1.0],  # Orange color (to distinguish from white ball)
        mass=0.45,  # 450g (standard soccer ball weight)
        friction=[0.8, 0.005, 0.0001],  # Ball-like friction
    )

    return spec


# Create the football entity configuration
# Initial position: penalty spot (11m from goal line)
# Goal is at y=-0.5m, so penalty spot is at y=10.5m (11m from goal)
FOOTBALL_BALL_ENTITY_CFG = EntityCfg(
    spec_fn=create_football_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 10.5, 0.11),  # Penalty spot: 11m from goal, resting on ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (no rotation)
        lin_vel=(0.0, 0.0, 0.0),  # No initial velocity (will be set by kick command)
    ),
)
