"""Ball entity configuration for goalkeeper training."""

import mujoco

from mjlab.entity import EntityCfg


def create_ball_spec() -> mujoco.MjSpec:
    """Create a ball (sphere) spec.

    This is a floating base, non-articulated entity that can move freely
    in 6-DOF. It's designed to be moved programmatically during episodes
    to simulate a moving ball for the goalkeeper to track/block.

    Returns:
        MjSpec: MuJoCo specification for the ball.
    """
    spec = mujoco.MjSpec()

    # Create a body with a freejoint (enables 6-DOF movement)
    body = spec.worldbody.add_body(name="ball_body")
    body.add_freejoint(name="ball_freejoint")

    # Add a sphere geometry (ball)
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.11, 0.11, 0.11],  # 22cm diameter (standard soccer ball)
        rgba=[1.0, 1.0, 1.0, 1.0],  # White color
        mass=0.45,  # 450g (standard soccer ball weight)
        friction=[0.8, 0.005, 0.0001],  # Ball-like friction
    )

    return spec


# Create the ball entity configuration
BALL_ENTITY_CFG = EntityCfg(
    spec_fn=create_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(10.0, 0.0, 0.11),  # Start on circle at radius=10m, resting on ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (no rotation)
        lin_vel=(0.0, 3.0, 0.0),  # Initial tangential velocity (v = ω*R = 0.3*10)
    ),
)
