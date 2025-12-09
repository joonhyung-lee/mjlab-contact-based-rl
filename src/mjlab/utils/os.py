import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def update_assets(
  assets: Dict[str, Any],
  path: Union[str, Path],
  meshdir: str | None = None,
  glob: str = "*",
  recursive: bool = False,
):
  """Update assets dictionary with files from a directory.

  This function reads files from a directory and adds them to an assets dictionary,
  with keys formatted to include the meshdir prefix when specified.

  Args:
    assets: Dictionary to update with file contents. Keys are asset paths, values are
      file contents as bytes.
    path: Path to directory containing asset files.
    meshdir: Optional mesh directory prefix, typically `spec.meshdir`. If provided,
      will be prepended to asset keys (e.g., "mesh.obj" becomes "custom_dir/mesh.obj").
    glob: Glob pattern for file matching. Defaults to "*" (all files).
    recursive: If True, recursively search subdirectories.
  """
  for f in Path(path).glob(glob):
    if f.is_file():
      asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
      assets[asset_key] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, meshdir, glob, recursive)


def dump_yaml(filename: Path, data: Dict, sort_keys: bool = False) -> None:
  """Saves data to a YAML file.

  Args:
      filename: The path to the YAML file.
      data: The data to save. Must be a dictionary.
      sort_keys: Whether to sort the keys in the YAML file.
  """
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(data, f, sort_keys=sort_keys)


def get_checkpoint_path(
  log_path: Path,
  run_dir: str = ".*",
  checkpoint: str = ".*",
  sort_alpha: bool = True,
) -> Path:
  """Get path to model checkpoint in input directory.

  The checkpoint file is resolved as: `<log_path>/<run_dir>/<checkpoint>`.

  If `run_dir` and `checkpoint` are regex expressions, then the most recent
  (highest alphabetical order) run and checkpoint are selected. To disable this
  behavior, set `sort_alpha` to `False`.
  """
  if not log_path.exists():
    raise ValueError(f"Log path does not exist: {log_path}")
  # Exclude wandb_checkpoints directory which is used for caching downloaded checkpoints.
  runs = [
    log_path / run.name
    for run in log_path.iterdir()
    if run.is_dir() and run.name != "wandb_checkpoints" and re.match(run_dir, run.name)
  ]
  if len(runs) == 0:
    raise ValueError(f"No run directories found in {log_path} matching '{run_dir}'")
  if sort_alpha:
    runs.sort()
  else:
    runs = sorted(runs, key=lambda p: p.stat().st_mtime)
  run_path = runs[-1]

  model_checkpoints = [
    f.name for f in run_path.iterdir() if re.match(checkpoint, f.name)
  ]
  if len(model_checkpoints) == 0:
    raise ValueError(f"No checkpoint found in {run_path} matching {checkpoint}")
  model_checkpoints.sort(key=lambda m: f"{m:0>15}")
  checkpoint_file = model_checkpoints[-1]
  return run_path / checkpoint_file


def construct_wandb_run_path(
  run_path: str, project: str | None = None, entity: str | None = None
) -> str:
  """Construct full wandb run path from partial or full path.

  Args:
    run_path: Wandb run path. Can be:
      - Full path: "entity/project/run_id"
      - Just run_id: "run_id" (requires project to be provided)
    project: Wandb project name. Required if run_path is just a run_id.
    entity: Wandb entity name. If None, will try to get from wandb API or environment.

  Returns:
    Full wandb run path in format "entity/project/run_id"
  """
  import os
  import wandb

  path_parts = run_path.split("/")

  # Check if we have a full path (entity/project/run_id) or just run_id
  if len(path_parts) >= 3:
    # Full path provided: entity/project/run_id
    return run_path
  elif len(path_parts) == 1:
    # Just run_id provided, need to construct full path
    run_id = path_parts[0]
    if project is None:
      raise ValueError(
        f"`wandb_run_path` '{run_path}' appears to be just a run_id. "
        "Please provide either:\n"
        "  1. Full path: 'entity/project/run_id', or\n"
        "  2. Just run_id with project name via agent config"
      )

    # Get entity from environment variable or wandb API
    if entity is None:
      entity = os.environ.get("WANDB_ENTITY")
      if entity is None:
        # Try to get from wandb API
        api = wandb.Api()
        try:
          # `viewer` is a property on newer wandb versions and a callable on older ones.
          viewer = api.viewer() if callable(api.viewer) else api.viewer
          entity = getattr(viewer, "username", None) or getattr(viewer, "login", None)
        except Exception:
          raise ValueError(
            "Could not determine wandb entity. Please set WANDB_ENTITY environment "
            f"variable or provide full path as 'entity/project/{run_id}'"
          )
        if entity is None:
          raise ValueError(
            "Could not determine wandb entity. Please set WANDB_ENTITY environment "
            f"variable or provide full path as 'entity/project/{run_id}'"
          )

    return f"{entity}/{project}/{run_id}"
  else:
    # Partial path (e.g., "project/run_id")
    raise ValueError(
      f"`wandb_run_path` '{run_path}' format is unclear. "
      "Please provide either:\n"
      "  1. Full path: 'entity/project/run_id', or\n"
      "  2. Just run_id: 'run_id' (requires project in config)"
    )


def get_wandb_checkpoint_path(
  log_path: Path, run_path: Path, project: str | None = None, entity: str | None = None
) -> tuple[Path, bool]:
  """Get checkpoint path from wandb, downloading if needed.

  Args:
    log_path: Local log directory path.
    run_path: Wandb run path. Can be:
      - Full path: "entity/project/run_id"
      - Just run_id: "run_id" (requires project and entity to be provided)
    project: Wandb project name. Required if run_path is just a run_id.
    entity: Wandb entity name. If None, will try to get from wandb API or environment.

  Returns:
    Tuple of (checkpoint_path, was_cached)
  """
  import wandb
  from wandb.errors import CommError

  run_path = Path(run_path)
  run_id = run_path.name

  def _latest_local_checkpoint(run_dir: Path) -> Path:
    """Return latest local model checkpoint in a run directory."""
    model_files: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
      if not p.is_file():
        continue
      m = re.match(r"model_(\d+)\.pt$", p.name)
      if m:
        model_files.append((int(m.group(1)), p))
    if not model_files:
      raise ValueError(
        f"No local model checkpoint files found in run directory: {run_dir}"
      )
    _, latest_path = max(model_files, key=lambda t: t[0])
    return latest_path

  # First, try to resolve a local run directory to support offline runs.
  local_run_dir: Path | None = None
  if run_path.is_dir():
    local_run_dir = run_path
  elif (log_path / run_id).is_dir():
    local_run_dir = log_path / run_id

  if local_run_dir is not None:
    checkpoint_path = _latest_local_checkpoint(local_run_dir)
    return checkpoint_path, True

  full_run_path = construct_wandb_run_path(str(run_path), project, entity)

  download_dir = log_path / "wandb_checkpoints" / run_id

  # Query wandb API to find the latest checkpoint.
  api = wandb.Api()
  try:
    wandb_run = api.run(full_run_path)
  except CommError as e:
    raise ValueError(
      f"Could not find wandb run '{full_run_path}'. "
      "If you want to load a local run, provide the run directory name instead."
    ) from e
  files = [file.name for file in wandb_run.files() if "model" in file.name]
  if not files:
    raise ValueError(f"No model checkpoint files found in wandb run: {full_run_path}")
  checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
  checkpoint_path = download_dir / checkpoint_file

  # If this checkpoint is not cached locally, download it.
  was_cached = checkpoint_path.exists()
  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_file = wandb_run.file(str(checkpoint_file))
    wandb_file.download(str(download_dir), replace=True)

  return checkpoint_path, was_cached
