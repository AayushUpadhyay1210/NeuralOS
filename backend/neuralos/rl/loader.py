"""Load a saved checkpoint as a scheduler, with clear errors when that is impossible."""

from __future__ import annotations

from collections.abc import Callable

from ..sim.schedulers import Scheduler
from . import registry
from .spaces import SpaceConfig


class ModelUnavailable(RuntimeError):
    """Raised when a model cannot be used; the message is safe to show to users."""


def load_scheduler(run_id: str, checkpoint: str = "best") -> Scheduler:
    missing = registry.missing_rl_dependencies()
    if missing:
        raise ModelUnavailable(
            "RL dependencies are not installed: " + ", ".join(missing)
            + ". Install them with: pip install -r backend/requirements-rl.txt"
        )
    meta = registry.metadata(run_id)
    if meta is None:
        raise ModelUnavailable(f"no trained model named '{run_id}' in {registry.run_dir(run_id)}")
    path = registry.checkpoint_path(run_id, checkpoint)
    if path is None and checkpoint == "best":
        path = registry.checkpoint_path(run_id, "final")
    if path is None:
        raise ModelUnavailable(
            f"model '{run_id}' has no saved checkpoint yet (status: {meta.get('status')})"
        )
    from .policy import RLScheduler, load_model

    model = load_model(path)
    space = SpaceConfig.from_dict(meta["space"])
    return RLScheduler(model, space, model_id=f"{run_id}:{path.stem}")


def scheduler_factory(checkpoint: str = "best") -> Callable[[str], Scheduler]:
    def make(run_id: str) -> Scheduler:
        return load_scheduler(run_id, checkpoint)

    return make
