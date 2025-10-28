# Goalkeeper Robot Environment

A custom mjlab environment for training a Unitree G1 humanoid robot as a goalkeeper.

## Run locally

To train the model locally, run the following command:

```bash
MUJOCO_GL=egl uv run train Mjlab-Goalkeeper --env.scene.num-envs 2048
```

To play (/test) the model locally, run the following command:

```bash
uv run play Mjlab-Goalkeeper-Play --wandb-run-path {your_wandb_username}/mjlab-goalkeeper/runs/{the_run_id}
```

## Run in the cloud (skypilot)

```bash
sky launch -d -c goalkeeper-task sky_task.yaml --secret WANDB_API_KEY=<your_wandb_api_key>
```
