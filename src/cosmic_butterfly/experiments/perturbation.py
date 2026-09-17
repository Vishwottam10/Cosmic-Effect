import numpy as np
from .scenarios import Scenario

def apply_velocity_perturbation(scenario: Scenario, body_index: int, delta_v: np.ndarray) -> Scenario:
    """
    Creates a new perturbed scenario by adding delta_v to a specific body's velocity.
    """
    perturbed = scenario.copy()
    perturbed.name = f"{scenario.name} (Perturbed)"
    perturbed.vel[body_index] += delta_v
    return perturbed
