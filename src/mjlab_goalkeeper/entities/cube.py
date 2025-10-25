"""Cube entity configuration for goalkeeper training."""

import mujoco

from mjlab.entity import EntityCfg


def create_cube_spec() -> mujoco.MjSpec:
    """Create a cube (box) spec.

    This is a floating base, non-articulated entity that can move freely
    in 6-DOF and interact with the robot and environment.

    Returns:
        MjSpec: MuJoCo specification for the cube.
    """
    spec = mujoco.MjSpec()

    # Create a body with a freejoint (enables 6-DOF movement)
    body = spec.worldbody.add_body(name="cube_body")
    body.add_freejoint(name="cube_freejoint")

    # Add a box geometry (cube)
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],  # 20cm cube
        rgba=[1.0, 0.3, 0.3, 1.0],  # Red color
        mass=0.5,  # 0.5 kg
        friction=[1.0, 0.005, 0.0001],  # Standard friction parameters
    )

    return spec


# Create the cube entity configuration
CUBE_ENTITY_CFG = EntityCfg(
    spec_fn=create_cube_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(2.0, 0.0, 0.5),  # 2 meters in front of robot, 0.5m above ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (no rotation)
    ),
)
