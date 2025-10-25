"""Goal entity configuration for goalkeeper training."""

import mujoco

from mjlab.entity import EntityCfg


def create_goal_spec() -> mujoco.MjSpec:
    """Create a simple soccer goal spec using boxes.

    Scaled-down goal dimensions for humanoid robot goalkeeper:
    - Width: 3.5m (reduced for humanoid scale)
    - Height: 2.0m (reduced for humanoid scale)
    - Post thickness: 0.12m (12cm, standard)

    The goal is positioned with the goal line at y=0, extending back in negative Y.
    Posts are at x = ±1.75m (half of 3.5m width).

    Returns:
        MjSpec: MuJoCo specification for the goal.
    """
    spec = mujoco.MjSpec()

    # Goal dimensions (scaled for humanoid robot)
    goal_width = 3.5  # meters (reduced from 7.32m FIFA standard)
    goal_height = 2.0  # meters (reduced from 2.44m)
    goal_depth = 1.0  # How far back the goal extends
    post_thickness = 0.12  # 12cm posts

    # Create a fixed body for the goal structure
    goal_body = spec.worldbody.add_body(name="goal_body")

    # Left post (vertical box on left side)
    goal_body.add_geom(
        name="goal_left_post",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[post_thickness / 2, post_thickness / 2, goal_height / 2],  # half-sizes
        pos=[-goal_width / 2, 0, goal_height / 2],  # Position at left side
        rgba=[1.0, 1.0, 1.0, 1.0],  # White
        friction=[1.0, 0.005, 0.0001],
    )

    # Right post (vertical box on right side)
    goal_body.add_geom(
        name="goal_right_post",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[post_thickness / 2, post_thickness / 2, goal_height / 2],  # half-sizes
        pos=[goal_width / 2, 0, goal_height / 2],  # Position at right side
        rgba=[1.0, 1.0, 1.0, 1.0],  # White
        friction=[1.0, 0.005, 0.0001],
    )

    # Crossbar (horizontal box connecting posts at the top)
    goal_body.add_geom(
        name="goal_crossbar",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[goal_width / 2, post_thickness / 2, post_thickness / 2],  # half-sizes
        pos=[0, 0, goal_height],  # Position at top center
        rgba=[1.0, 1.0, 1.0, 1.0],  # White
        friction=[1.0, 0.005, 0.0001],
    )

    # Back net (thin box at the back)
    goal_body.add_geom(
        name="goal_back_net",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[
            goal_width / 2,
            0.05,
            goal_height / 2,
        ],  # half-width, thin depth, half-height
        pos=[0, -goal_depth, goal_height / 2],
        rgba=[0.9, 0.9, 0.9, 0.6],  # Semi-transparent white
        friction=[0.5, 0.005, 0.0001],
    )

    # Ground bar (optional, helps define goal area)
    goal_body.add_geom(
        name="goal_ground_bar",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[goal_width / 2, goal_depth / 2, post_thickness / 4],  # Thin ground marker
        pos=[0, -goal_depth / 2, post_thickness / 4],
        rgba=[0.8, 0.8, 0.8, 0.8],  # Light gray
        friction=[1.0, 0.005, 0.0001],
    )

    return spec


# Create the goal entity configuration
# This is a fixed-base, non-articulated entity (doesn't move)
GOAL_ENTITY_CFG = EntityCfg(
    spec_fn=create_goal_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, -0.5, 0.0),  # Goal line slightly behind robot (at y=-0.5m)
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (no rotation)
    ),
)
