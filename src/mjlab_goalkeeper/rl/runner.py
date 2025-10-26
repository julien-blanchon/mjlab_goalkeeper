from pathlib import Path

import wandb
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab_goalkeeper.rl.exporter import (
    attach_onnx_metadata,
    export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(OnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = (Path(policy_path).stem + ".onnx").name
            if self.alg.policy.actor_obs_normalization:
                normalizer = self.alg.policy.actor_obs_normalizer
            else:
                normalizer = None
            export_velocity_policy_as_onnx(
                self.alg.policy,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped,
                wandb.run.name,  # type: ignore
                path=policy_path,
                filename=filename,
            )
            wandb.save(policy_path + filename, base_path=Path(policy_path).parent)
