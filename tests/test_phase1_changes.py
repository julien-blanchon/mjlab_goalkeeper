#!/usr/bin/env python3
"""Test Phase 1 reward changes are working correctly."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_goalkeeper.config.env_cfg import UnitreeG1GoalkeeperEnvCfg


def test_phase1_changes():
    """Verify Phase 1 reward configuration changes."""
    print("=" * 70)
    print("Testing Phase 1 Reward Changes")
    print("=" * 70)

    # Create environment
    print("\n[1/5] Creating environment...")
    env_cfg = UnitreeG1GoalkeeperEnvCfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    print("✓ Environment created")

    # Check reward configuration
    print("\n[2/5] Verifying reward configuration...")
    reward_manager = env.reward_manager

    # Check alive weight
    alive_term = reward_manager._terms["alive"]
    alive_weight = reward_manager._term_cfgs["alive"].weight
    print(f"  alive weight: {alive_weight}")
    assert alive_weight == 0.4, f"Expected alive weight=0.4, got {alive_weight}"
    print("  ✓ alive weight reduced to 0.4 (from 1.0)")

    # Check hand_to_ball std
    hand_to_ball_cfg = reward_manager._term_cfgs["hand_to_ball"]
    hand_to_ball_std = hand_to_ball_cfg.params["std"]
    print(f"  hand_to_ball std: {hand_to_ball_std}")
    assert hand_to_ball_std == 1.5, f"Expected std=1.5, got {hand_to_ball_std}"
    print("  ✓ hand_to_ball std increased to 1.5 (from 0.3)")

    # Check air_time is enabled
    air_time_weight = reward_manager._term_cfgs["air_time"].weight
    print(f"  air_time weight: {air_time_weight}")
    assert air_time_weight == 0.5, (
        f"Expected air_time weight=0.5, got {air_time_weight}"
    )
    assert "air_time" in reward_manager.active_terms, "air_time should be active"
    print("  ✓ air_time enabled with weight 0.5")

    # Reset and take a step
    print("\n[3/5] Testing reward computation...")
    obs, info = env.reset()
    action = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Total reward: {reward[0]:.4f}")
    print("  ✓ Rewards computed without errors")

    # Check individual reward values
    print("\n[4/5] Checking individual reward values...")
    robot = env.scene["robot"]
    football = env.scene["football_ball"]

    robot_pos = robot.data.root_link_pos_w[0]
    ball_pos = football.data.root_link_pos_w[0]
    distance = torch.norm(ball_pos - robot_pos).item()

    print(f"  Robot position: {robot_pos}")
    print(f"  Ball position: {ball_pos}")
    print(f"  Distance to ball: {distance:.2f}m")

    # Check that hand_to_ball gives non-zero reward even at long distance
    from mjlab_goalkeeper.mdp.rewards import hand_to_ball_distance

    hand_reward = hand_to_ball_distance(
        env,
        std=1.5,
        left_hand_body="left_wrist_yaw_link",
        right_hand_body="right_wrist_yaw_link",
    )
    print(f"  hand_to_ball reward at {distance:.1f}m: {hand_reward[0]:.6f}")

    if distance > 5.0:
        assert hand_reward[0] > 1e-6, (
            f"hand_to_ball should give non-zero reward at {distance:.1f}m! "
            f"Got {hand_reward[0]:.6e}"
        )
        print(f"  ✓ hand_to_ball gives meaningful gradient at {distance:.1f}m")
    else:
        print(f"  ℹ Ball is close ({distance:.1f}m), gradient check skipped")

    # Compare with old std=0.3 for demonstration
    old_std_reward = torch.exp(-distance / 0.3**2)
    new_std_reward = torch.exp(-distance / 1.5**2)
    print(f"\n  Comparison at {distance:.1f}m distance:")
    print(f"    Old (std=0.3): reward = {old_std_reward:.8f}")
    print(f"    New (std=1.5): reward = {new_std_reward:.8f}")
    print(f"    Improvement: {new_std_reward / old_std_reward:.1f}x stronger signal!")

    # Test gradient at various distances
    print("\n[5/5] Testing reward gradient at various distances...")
    print("  Distance | Old (std=0.3) | New (std=1.5) | Ratio")
    print("  " + "-" * 60)
    for dist in [0.5, 1.0, 2.0, 5.0, 10.0]:
        old_r = torch.exp(torch.tensor(-dist / 0.3**2))
        new_r = torch.exp(torch.tensor(-dist / 1.5**2))
        ratio = (new_r / old_r).item()
        print(f"  {dist:4.1f}m   | {old_r:.6e}    | {new_r:.6e}    | {ratio:6.1f}x")

    print("\n  ✓ New std provides much stronger gradient at long range!")
    print("    (This solves the 'gradient desert' problem)")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL PHASE 1 CHANGES VERIFIED!")
    print("=" * 70)
    print("\nChanges Applied:")
    print("  1. alive reward: 1.0 → 0.4 (reduces 'standing still' bias)")
    print("  2. hand_to_ball std: 0.3 → 1.5 (fixes long-range gradient)")
    print("  3. air_time: enabled with weight 0.5 (teaches locomotion)")
    print("\nExpected Training Improvements:")
    print("  • Robot should now 'feel' the ball at 10m distance")
    print("  • Robot should start experimenting with movement")
    print("  • Locomotion becomes a viable strategy")
    print("\nNext Step: Start training and monitor for movement behavior!")
    print("  Command: ./train.py Mjlab-Goalkeeper --env.scene.num-envs 4096")
    print()


if __name__ == "__main__":
    test_phase1_changes()
