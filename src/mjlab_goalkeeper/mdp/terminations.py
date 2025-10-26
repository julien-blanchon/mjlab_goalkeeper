"""Custom termination functions for goalkeeper environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv


def goal_scored_termination(env: ManagerBasedEnv) -> torch.Tensor:
    """Termination function for goal scored (returns boolean per env).

    Args:
        env: The environment.

    Returns:
        Boolean tensor of shape (num_envs,). True if goal scored.
    """
    # Import here to use observation function
    from mjlab_goalkeeper.mdp.observations import goal_scored_detection

    return goal_scored_detection(env).squeeze(-1) > 0.5
