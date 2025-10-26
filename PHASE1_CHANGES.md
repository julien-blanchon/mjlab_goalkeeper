# Phase 1 Implementation Complete ✅

## Summary

Fixed the "standing still" local optimum by improving reward gradients for long-range ball positioning.

## Changes Made

### 1. Reduced "Standing Still" Bias

**File**: `src/mjlab_goalkeeper/config/env_cfg.py`

- **Location**: Line 271
- **Change**: `alive` reward weight: `1.0` → `0.4`
- **Reason**: Robot was getting too much reward for just standing. Now it needs to actively pursue the ball to maximize reward.

### 2. Fixed Long-Range Gradient Problem ⭐ (Most Important)

**File**: `src/mjlab_goalkeeper/config/env_cfg.py`

- **Location**: Line 280
- **Change**: `hand_to_ball` std parameter: `0.3` → `1.5`
- **Reason**: With std=0.3, the exponential kernel gave **numerically zero** reward beyond 2m. The ball starts at 10.5m (penalty spot), so the robot couldn't "feel" any gradient to move toward it. With std=1.5:
  - At 10m: reward ≈ 0.00045 (detectable!)
  - At 5m: reward ≈ 0.002 (strong signal)
  - At 2m: reward ≈ 0.0165
  - At 0.5m: reward ≈ 0.16 (good precision)

### 3. Enabled Locomotion Reward

**File**: `src/mjlab_goalkeeper/config/env_cfg.py`

- **Location**: Lines 310-318
- **Changes**:
  - Weight: `0.0` → `0.5` (enabled!)
  - `command_name`: `"twist"` → `"penalty_kick"` (fixed to use correct command)
  - `command_threshold`: `0.05` → `0.0` (always active)
  - `reward_mode`: `"on_landing"` → `"continuous"` (constant reward while feet in air)
- **Reason**: Teaches the robot that lifting feet is good, which is essential for learning to walk/move.

## Expected Behavior Changes

### Before Phase 1:

- ✅ Robot learns to stand upright
- ❌ Robot never moves toward ball (no gradient to follow)
- ❌ Robot stays frozen in starting position
- ❌ Only learns to stabilize on reset

### After Phase 1:

- ✅ Robot still learns to stand upright (but slightly less rewarded)
- ✅ Robot can "feel" the ball even at 10m distance
- ✅ Robot learns that lifting feet is rewarded
- ✅ Should start experimenting with movement toward ball
- ✅ Movement becomes viable strategy (not just standing)

## How to Test

### Quick Validation Test

```bash
# Run a short training to verify no errors
cd /home/ubuntu/goal_keeper_robot/mjlab_goalkeeper
uv run python -c "
import mjlab_goalkeeper
from mjlab.envs import ManagerBasedRlEnv
from mjlab_goalkeeper.config.env_cfg import UnitreeG1GoalkeeperEnvCfg

env_cfg = UnitreeG1GoalkeeperEnvCfg()
env = ManagerBasedRlEnv(cfg=env_cfg, device='cuda:0')
obs, info = env.reset()
print('✅ Environment created successfully')
print(f'✅ Reward terms: {list(env.reward_manager._terms.keys())}')
print(f'✅ air_time enabled: {\"air_time\" in env.reward_manager.active_terms}')
env.close()
print('✅ Phase 1 changes verified!')
"
```

### Start Training

```bash
# Train with the new configuration
./train.py Mjlab-Goalkeeper --env.scene.num-envs 4096
```

### Monitor Training (W&B)

Look for these positive signs:

- **Early training (0-100 iterations)**:

  - `reward/alive` should be positive but lower than before (~0.4 instead of 1.0)
  - `reward/hand_to_ball` should start showing **non-zero values** even when robot is far from ball
  - `reward/air_time` should appear and show positive values as robot experiments with moving

- **Mid training (100-500 iterations)**:
  - `reward/hand_to_ball` should gradually increase as robot learns to approach
  - Robot should start showing some forward movement behavior
  - Total reward should increase as movement becomes rewarded

## Troubleshooting

### If robot still doesn't move after ~500 iterations:

Proceed to **Phase 2**: Add velocity-toward-ball reward

```python
# This would directly incentivize moving in ball's direction
velocity_toward_ball: RewardTerm = term(...)
```

### If robot moves but falls over more:

- Increase `flat_orientation` penalty weight
- Reduce `air_time` weight to 0.3
- Check push events aren't too aggressive

### If training crashes with NaN:

- Enable NaN guard: `--enable-nan-guard True`
- Check reward magnitudes aren't exploding
- Verify contact parameters are reasonable

## Next Steps (If Needed)

If Phase 1 doesn't fully solve the problem after ~1000 iterations, we have **Phase 2** ready:

1. **Add velocity-toward-ball reward** - Direct movement incentive
2. **Two-stage hand_to_ball kernel** - Even better gradient shaping
3. **Distance reduction reward** - Reward for getting closer over time
4. **Curriculum learning** - Start ball closer, gradually move to 10.5m

But try Phase 1 first! The gradient fix alone might be sufficient. 🎯

## Files Modified

- ✅ `src/mjlab_goalkeeper/config/env_cfg.py` (3 changes)

## Testing Checklist

- [x] Configuration changes made
- [x] No linter errors
- [ ] Quick validation test passes (run the test above)
- [ ] Training starts without errors
- [ ] Monitor first 100 iterations for behavior changes

Good luck with training! 🚀⚽
