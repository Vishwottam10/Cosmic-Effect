import numpy as np
import taichi as ti
from cosmic_butterfly.physics.simulation import Simulation
from cosmic_butterfly.experiments.scenarios import (
    create_pythagorean_three_body,
    create_hierarchical_three_body,
    create_figure_eight
)
from cosmic_butterfly.experiments.perturbation import apply_velocity_perturbation
from cosmic_butterfly.experiments.divergence import calculate_divergence

ti.init(arch=ti.cpu)

def analyze_scenario(scenario_func, name, steps, print_interval=1000):
    print(f"\n--- Analyzing {name} ---", flush=True)
    
    baseline = scenario_func()
    # Apply a tiny perturbation of 1e-5 to body 1's x-velocity
    delta_v = np.array([1e-5, 0.0], dtype=np.float64)
    perturbed = apply_velocity_perturbation(baseline, body_index=1, delta_v=delta_v)
    
    sim_a = Simulation(baseline.num_bodies)
    sim_a.set_state(baseline.pos, baseline.vel, baseline.mass)
    
    sim_b = Simulation(perturbed.num_bodies)
    sim_b.set_state(perturbed.pos, perturbed.vel, perturbed.mass)
    
    dt = baseline.dt
    
    initial_d = calculate_divergence(sim_a, sim_b)
    print(f"Initial Divergence D(0): {initial_d:.2e}")
    
    max_d = 0.0
    
    for step in range(steps):
        sim_a.step(dt)
        sim_b.step(dt)
        
        d = calculate_divergence(sim_a, sim_b)
        if d > max_d:
            max_d = d
            
        if (step + 1) % print_interval == 0:
            print(f"Step {step+1}: D(t) = {d:.2e}")
            
    print(f"Final Divergence D(t_end): {d:.2e}")
    print(f"Max Divergence: {max_d:.2e}")

if __name__ == "__main__":
    # Test Pythagorean 3-body (highly chaotic)
    analyze_scenario(create_pythagorean_three_body, "Pythagorean 3-body", steps=5000, print_interval=1000)
    
    # Test Figure-8 (stable)
    analyze_scenario(create_figure_eight, "Figure-8", steps=5000, print_interval=1000)
    
    # Test Hierarchical (mostly stable, potentially secularly chaotic over long times)
    analyze_scenario(create_hierarchical_three_body, "Hierarchical 3-body", steps=5000, print_interval=1000)
