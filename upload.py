import wandb, os
WANDB_ENTITY="rebel-joonhyung-lee-rebellions"
WANDB_PROJECT="motion-upload"

run=wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="upload-motion")
artifact=wandb.Artifact(name="motion", type="motions")
artifact.add_file("motion.npz")
logged=run.log_artifact(artifact)
run.link_artifact(logged, target_path=f"wandb-registry-motions/{artifact.name}")
"""
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name rebel-joonhyung-lee-rebellions/motion-upload/motion:v0 --env.scene.num-envs 4096
"""