"""Training: one Trainer (mechanics) + Tasks (objectives)."""

from .tasks import Task, DirectTask, ARTask, build_task
from .trainer import Trainer

__all__ = ["Task", "DirectTask", "ARTask", "build_task", "Trainer"]
