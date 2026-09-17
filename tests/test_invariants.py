import numpy as np
import pytest
import taichi as ti
from cosmic_butterfly.physics.constants import G
from cosmic_butterfly.physics.simulation import Simulation
from cosmic_butterfly.physics.invariants import (
    compute_total_energy,
    compute_linear_momentum,
    compute_angular_momentum
)

ti.init(arch=ti.cpu)

def setup_system():
    # 3-body system
    pos = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    
    vel = np.array([
        [0.0, 0.5],
        [0.0, -0.5],
        [0.5, 0.0]
    ], dtype=np.float64)
    
    mass = np.array([1.0, 1.0, 0.5], dtype=np.float64)
    
    sim = Simulation(3)
    sim.set_state(pos, vel, mass)
    return sim

def test_energy_conservation():
    sim = setup_system()
    dt = 0.001
    
    initial_energy = compute_total_energy(sim.n, sim.pos, sim.vel, sim.mass)
    
    for _ in range(100):
        sim.step(dt)
        
    final_energy = compute_total_energy(sim.n, sim.pos, sim.vel, sim.mass)
    
    # Energy error should be very small for symplectic integrator over short time
    rel_error = abs((final_energy - initial_energy) / initial_energy)
    assert rel_error < 1e-4

def test_momentum_conservation():
    sim = setup_system()
    dt = 0.001
    
    initial_p = compute_linear_momentum(sim.n, sim.vel, sim.mass).to_numpy()
    
    for _ in range(100):
        sim.step(dt)
        
    final_p = compute_linear_momentum(sim.n, sim.vel, sim.mass).to_numpy()
    
    # Linear momentum should be conserved to machine precision
    assert np.allclose(initial_p, final_p, atol=1e-10)

def test_angular_momentum_conservation():
    sim = setup_system()
    dt = 0.001
    
    initial_L = compute_angular_momentum(sim.n, sim.pos, sim.vel, sim.mass)
    
    for _ in range(100):
        sim.step(dt)
        
    final_L = compute_angular_momentum(sim.n, sim.pos, sim.vel, sim.mass)
    
    assert abs(final_L - initial_L) < 1e-10
