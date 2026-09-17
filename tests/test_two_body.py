import numpy as np
import pytest
import taichi as ti
from cosmic_butterfly.physics.constants import G
from cosmic_butterfly.physics.simulation import Simulation

ti.init(arch=ti.cpu)

def test_two_body_orbit():
    """
    Test two-body problem:
    Mass 1 = M at origin
    Mass 2 = m at distance R
    v = sqrt(G*M/R)
    Period T = 2*pi*sqrt(R^3 / (G*M))
    """
    M = 1.0
    R = 1.0
    v = np.sqrt(G * M / R)
    
    pos = np.array([
        [0.0, 0.0],
        [R, 0.0]
    ], dtype=np.float64)
    
    # Body 2 moving in y direction
    vel = np.array([
        [0.0, 0.0],
        [0.0, v]
    ], dtype=np.float64)
    
    mass = np.array([M, 1e-6], dtype=np.float64)
    
    sim = Simulation(2)
    sim.set_state(pos, vel, mass)
    
    # Period T
    T = 2 * np.pi * np.sqrt(R**3 / (G * M))
    
    dt = 0.001
    steps = int(T / dt)
    
    initial_pos, _, _ = sim.get_state()
    
    for _ in range(steps):
        sim.step(dt)
        
    final_pos, _, _ = sim.get_state()
    
    # After one period, body 2 should be back at [R, 0]
    dist_error = np.linalg.norm(final_pos[1] - initial_pos[1])
    assert dist_error < 0.01  # 1% error is acceptable for this dt
