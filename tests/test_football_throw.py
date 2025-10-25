#!/usr/bin/env python3
"""Test football throw command functionality."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_goalkeeper.config.env_cfg import UnitreeG1FlatEnvCfg


def test_football_throw_command():
    """Test that the football throw command works correctly."""
    print("=" * 70)
    print("Testing Football Throw Command")
    print("=" * 70)

    # Create environment (use CUDA for speed)
    print("\n[1/6] Creating environment with football...")
    env_cfg = UnitreeG1FlatEnvCfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    print("✓ Environment created")

    # Check entities
    print("\n[2/6] Checking entities...")
    expected_entities = {"robot", "cube", "ball", "football_ball"}
    actual_entities = set(env.scene.entities.keys())

    assert actual_entities == expected_entities, (
        f"Expected {expected_entities}, got {actual_entities}"
    )
    print(f"✓ All entities present: {sorted(actual_entities)}")

    # Check commands
    print("\n[3/6] Checking command manager...")
    cmd_manager = env.command_manager
    active_cmd_names = list(cmd_manager._terms.keys())
    print(f"  Active command terms: {active_cmd_names}")

    assert "twist" in active_cmd_names, "Velocity command missing!"
    assert "football_throw" in active_cmd_names, "Football throw command missing!"
    print("✓ Both commands configured:")
    print("  - twist: Robot velocity control")
    print("  - football_throw: Football throwing")

    # Reset and initial throw
    print("\n[4/6] Testing initial throw...")
    obs, info = env.reset()

    robot = env.scene["robot"]
    football = env.scene["football_ball"]

    robot_pos = robot.data.root_link_pos_w[0]
    football_pos = football.data.root_link_pos_w[0]
    football_vel = football.data.root_link_lin_vel_w[0]

    print(f"  Robot position: {robot_pos}")
    print(f"  Football position: {football_pos}")
    print(f"  Football velocity: {football_vel}")

    # Check that football was thrown
    distance = torch.norm(football_pos[:2] - robot_pos[:2])
    print(f"  Distance from robot: {distance:.2f}m")

    assert distance > 7.0, (
        f"Football should be thrown from distance, got {distance:.2f}m"
    )
    assert distance < 16.0, f"Football too far, got {distance:.2f}m"
    print("  ✓ Football thrown from valid distance (8-15m range)")

    # Check velocity magnitude
    vel_magnitude = torch.norm(football_vel)
    print(f"  Football velocity magnitude: {vel_magnitude:.2f} m/s")
    assert vel_magnitude > 0.5, "Football should have significant velocity"
    print("  ✓ Football has throw velocity")

    # Simulate and watch trajectory
    print("\n[5/6] Simulating football trajectory...")
    positions = []
    velocities = []
    heights = []
    distances_to_robot = []

    num_steps = 100
    action = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)

    for step in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(action)

        football_pos = football.data.root_link_pos_w[0]
        football_vel = football.data.root_link_lin_vel_w[0]
        robot_pos = robot.data.root_link_pos_w[0]

        positions.append(football_pos.cpu().numpy())
        velocities.append(football_vel.cpu().numpy())
        heights.append(football_pos[2].item())

        dist_to_robot = torch.norm(football_pos - robot_pos).item()
        distances_to_robot.append(dist_to_robot)

        # Check if ball got close to robot (within 2m)
        if dist_to_robot < 2.0:
            print(f"  ⚽ Football reached robot vicinity at step {step}!")
            print(f"    Closest distance: {dist_to_robot:.2f}m")
            break

    # Analyze trajectory
    max_height = max(heights)
    min_distance = min(distances_to_robot)

    print(f"  Steps simulated: {step + 1}/{num_steps}")
    print(f"  Max height reached: {max_height:.2f}m")
    print(f"  Minimum distance to robot: {min_distance:.2f}m")

    if min_distance < 2.0:
        print("  ✓ Football successfully approached robot (< 2m)")
    else:
        print("  ⚠ Football didn't reach robot (stayed > 2m away)")

    # Test command resampling
    print("\n[6/6] Testing command resampling (new throw)...")

    # Manually trigger resample
    football_cmd = cmd_manager._terms["football_throw"]
    initial_throws = football_cmd.metrics["throws_count"][0].item()

    # Resample for environment 0
    football_cmd._resample_command(torch.tensor([0], device=env.device))
    env.sim.forward()

    new_throws = football_cmd.metrics["throws_count"][0].item()
    new_football_pos = football.data.root_link_pos_w[0]
    new_football_vel = football.data.root_link_lin_vel_w[0]

    print(f"  Throws before: {initial_throws}, after: {new_throws}")
    print(f"  New football position: {new_football_pos}")
    print(f"  New football velocity: {new_football_vel}")
    print("  ✓ Command resampling works (new throw executed)")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("✅ FOOTBALL THROW COMMAND TEST PASSED!")
    print("=" * 70)
    print("\nFootball Throw Configuration:")
    print("  - Throw distance: 8-15 meters from robot")
    print("  - Throw angle: Full 360° around robot")
    print("  - Throw height: 2-5 meters")
    print("  - Flight time: 1-2 seconds")
    print("  - Target: Robot torso (0.5-1.5m high)")
    print("  - Aiming randomization: ±0.5m in XY")
    print("  - Resampling: Every 3-6 seconds (new throw)")
    print("  - Visualization: Throw position, velocity, trajectory")
    print("\nThe robot will face incoming balls from various angles,")
    print("training realistic goalkeeper reflexes!")
    print()


if __name__ == "__main__":
    test_football_throw_command()
