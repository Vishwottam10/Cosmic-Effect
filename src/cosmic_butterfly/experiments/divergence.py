import numpy as np
from cosmic_butterfly.physics.simulation import Simulation

def calculate_divergence(sim_a: Simulation, sim_b: Simulation) -> float:
    """
    Calculate Euclidean separation metric D(t) between corresponding bodies.
    D(t) = sqrt(sum_i |r_i^A - r_i^B|^2)
    """
    pos_a, _, _ = sim_a.get_state()
    pos_b, _, _ = sim_b.get_state()
    
    diff = pos_a - pos_b
    d = np.sqrt(np.sum(diff**2))
    return float(d)
