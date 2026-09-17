import argparse
import numpy as np
import taichi as ti
import time

from cosmic_butterfly.physics.simulation import Simulation
from cosmic_butterfly.physics.invariants import compute_total_energy
from cosmic_butterfly.experiments.scenarios import (
    create_pythagorean_three_body,
    create_hierarchical_three_body,
    create_figure_eight,
)
from cosmic_butterfly.experiments.perturbation import apply_velocity_perturbation
from cosmic_butterfly.experiments.divergence import calculate_divergence
from cosmic_butterfly.visualization.renderer import Renderer, VisMode

def main():
    parser = argparse.ArgumentParser(description="Cosmic Butterfly Effect")
    parser.add_argument(
        "--scenario", type=str, default="pythagorean",
        choices=["pythagorean", "hierarchical", "figure8"],
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    args = parser.parse_args()

    if args.gpu:
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)

    print(f"Starting Cosmic Butterfly Effect - {args.scenario} scenario")

    # ---- 1. Build scenario ------------------------------------------------
    if args.scenario == "pythagorean":
        baseline = create_pythagorean_three_body()
    elif args.scenario == "hierarchical":
        baseline = create_hierarchical_three_body()
    else:
        baseline = create_figure_eight()

    # ---- 2. Initialize Twins ----------------------------------------------
    dv_mag = 1e-5
    
    sim_a = Simulation(baseline.num_bodies)
    sim_b = Simulation(baseline.num_bodies)
    
    def reset_simulations(dv):
        delta_v = np.array([dv, 0.0], dtype=np.float64)
        perturbed = apply_velocity_perturbation(baseline, body_index=1, delta_v=delta_v)
        
        sim_a.set_state(baseline.pos, baseline.vel, baseline.mass)
        sim_b.set_state(perturbed.pos, perturbed.vel, perturbed.mass)
        
        return compute_total_energy(sim_a.n, sim_a.pos, sim_a.vel, sim_a.mass)

    initial_energy_a = reset_simulations(dv_mag)

    # ---- 3. Renderer & Cinematic State ------------------------------------
    renderer = Renderer(title=f"Cosmic Butterfly Effect - {baseline.name}")
    renderer.perturbation_val = dv_mag
    
    if args.scenario == "pythagorean":
        steps_per_frame = 40
        trail_record_interval = 4  # Record trail every 4 physics steps
    else:
        steps_per_frame = 20
        trail_record_interval = 2

    # Cinematic Phases
    PHASE_1_INITIAL = 1
    PHASE_2_RELEASE = 2
    PHASE_3_DIVERGENCE = 3
    PHASE_4_OUTCOME = 4
    
    current_phase = PHASE_1_INITIAL
    phase_start_time = time.time()
    
    phase_names = {
        PHASE_1_INITIAL: "1. INITIAL CONDITIONS",
        PHASE_2_RELEASE: "2. RELEASE",
        PHASE_3_DIVERGENCE: "3. DIVERGENCE",
        PHASE_4_OUTCOME: "4. OUTCOME"
    }

    last_dv_mag = dv_mag
    
    print("Simulation running. Close the window to exit.")

    # ---- 4. Main loop -----------------------------------------------------
    while renderer.is_running():
        # Handle UI Changes
        if renderer.perturbation_val != last_dv_mag:
            last_dv_mag = renderer.perturbation_val
            initial_energy_a = reset_simulations(last_dv_mag)
            renderer.clear_trails()
            renderer.reset_graph()
            current_phase = PHASE_1_INITIAL
            phase_start_time = time.time()

        now = time.time()
        elapsed_in_phase = now - phase_start_time

        # State Machine Transitions
        if current_phase == PHASE_1_INITIAL:
            if elapsed_in_phase > 4.0:
                current_phase = PHASE_2_RELEASE
                phase_start_time = now
        elif current_phase == PHASE_2_RELEASE:
            if elapsed_in_phase > 2.0:
                current_phase = PHASE_3_DIVERGENCE
                phase_start_time = now

        # Physics update
        if current_phase >= PHASE_2_RELEASE:
            # Slow down if in outcome phase to emphasize end
            actual_steps = steps_per_frame
            if current_phase == PHASE_4_OUTCOME:
                actual_steps = steps_per_frame // 4

            for step in range(actual_steps):
                sim_a.step(baseline.dt)
                sim_b.step(baseline.dt)
                
                # Decoupled trail recording! High frequency sampling directly from physics
                if step % trail_record_interval == 0:
                    renderer.record_trail_step(sim_a, sim_b)

        # Diagnostics
        d = calculate_divergence(sim_a, sim_b)
        if current_phase == PHASE_3_DIVERGENCE and d > 0.5:
            current_phase = PHASE_4_OUTCOME
            phase_start_time = now

        cur_energy = compute_total_energy(sim_a.n, sim_a.pos, sim_a.vel, sim_a.mass)
        energy_err = abs((cur_energy - initial_energy_a) / initial_energy_a) if initial_energy_a != 0 else 0.0

        # Handle camera input
        renderer.handle_input()

        # Draw
        renderer.render(
            sim_a, sim_b, 
            draw_b=True, 
            sim_time=sim_a.time, 
            div=d
        )
        renderer.draw_hud(
            sim_time=sim_a.time,
            divergence=d,
            energy_err=energy_err,
            scenario_name=baseline.name,
            phase_name=phase_names[current_phase]
        )
        
        # Outcome overlay
        if current_phase == PHASE_4_OUTCOME:
            with renderer.gui.sub_window("CONCLUSION", 0.35, 0.4, 0.3, 0.2) as w:
                w.text("Initial conditions were nearly identical.")
                w.text("Gravitational interactions amplified differences.")
                w.text("Result: Dramatically different futures.")
                
                if w.button("Restart Experiment"):
                    initial_energy_a = reset_simulations(last_dv_mag)
                    renderer.clear_trails()
                    renderer.reset_graph()
                    current_phase = PHASE_1_INITIAL
                    phase_start_time = time.time()
                
        renderer.show()

if __name__ == "__main__":
    main()
