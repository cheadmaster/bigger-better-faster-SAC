"""Utilities for scheduling target-network Polyak updates."""

import math


_VALID_SCHEDULES = ("constant", "linear", "cosine")


def _validate_tau(name, value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return value


def create_tau_scheduler(
    initial_tau,
    final_tau=None,
    schedule="constant",
    schedule_steps=None,
):
    """Builds a bounded target-network update-rate scheduler.

    Non-constant schedules increase ``tau`` from ``initial_tau`` to
    ``final_tau`` over ``schedule_steps`` gradient steps. Steps outside the
    schedule range are clipped, so the returned value always remains in the
    validated interval. A constant schedule exactly preserves the original
    target-network update behavior.

    Args:
      initial_tau: Initial Polyak update rate in [0, 1].
      final_tau: Final update rate. Required for non-constant schedules and
        must be greater than or equal to ``initial_tau``.
      schedule: One of ``constant``, ``linear``, or ``cosine``.
      schedule_steps: Number of gradient steps in a non-constant schedule.

    Returns:
      A callable mapping a gradient-step count to a Python float.

    Raises:
      ValueError: If the schedule configuration is invalid.
    """
    initial_tau = _validate_tau("initial_tau", initial_tau)
    schedule = str(schedule).lower()
    if schedule not in _VALID_SCHEDULES:
        raise ValueError(
            f"Unknown target update schedule {schedule!r}; "
            f"expected one of {_VALID_SCHEDULES}")

    final_tau_was_none = final_tau is None
    if final_tau_was_none:
        final_tau = initial_tau
    final_tau = _validate_tau("final_tau", final_tau)

    if schedule == "constant":
        if final_tau != initial_tau:
            raise ValueError(
                "final_tau must equal initial_tau for a constant schedule")
        return lambda step: initial_tau

    if final_tau_was_none:
        raise ValueError("final_tau is required for a non-constant schedule")
    if final_tau < initial_tau:
        raise ValueError(
            "final_tau must be greater than or equal to initial_tau")
    if schedule_steps is None or int(schedule_steps) <= 0:
        raise ValueError(
            "schedule_steps must be positive for a non-constant schedule")
    schedule_steps = int(schedule_steps)

    def scheduler(step):
        progress = min(max(float(step) / schedule_steps, 0.0), 1.0)
        if schedule == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return initial_tau + (final_tau - initial_tau) * progress

    return scheduler


def polyak_update(target_params, online_params, tau):
    """Interpolates matching target and online parameter PyTrees.

    JAX is imported lazily so schedule validation remains usable in lightweight
    environments that do not have the training dependencies installed. Inside
    the jitted training step, ``tau`` is a scalar placed on the selected JAX
    device together with the parameter leaves.
    """
    import jax  # pylint: disable=g-import-not-at-top

    return jax.tree_util.tree_map(
        lambda target, online: target * (1.0 - tau) + online * tau,
        target_params,
        online_params,
    )
