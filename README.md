# Cosmic Butterfly Effect

*"How a microscopic change in initial conditions can rewrite the future of a planetary system."*

## Overview

**Cosmic Butterfly Effect** is a competition-grade interactive scientific simulation that demonstrates the sensitive dependence on initial conditions (the "Butterfly Effect") in gravitational N-body dynamics.

It is built as a highly accurate computational experiment. It runs two identical universes side by side, where the only difference is a microscopic velocity perturbation in one body. 
The visualization is driven entirely by the real-time symplectic numerical integration of Newton's laws of gravity.

## The Scientific Experiment

### The Physics Model
We simulate Newtonian gravitational N-body dynamics using a Velocity Verlet (kick-drift-kick) symplectic integrator. This integrator is chosen because it perfectly conserves the phase-space volume and exhibits excellent long-term energy stability, which is crucial for distinguishing genuine chaotic divergence from numerical drift.

The gravitational acceleration for each body $i$ is given by:
$$ \vec{a}_i = G \sum_{j \neq i} m_j \frac{\vec{r}_j - \vec{r}_i}{(|\vec{r}_j - \vec{r}_i|^2 + \epsilon^2)^{3/2}} $$
where $\epsilon = 10^{-6}$ is a softening parameter used to prevent unphysical singularities during close encounters.

### Unit System
The simulation uses normalized computational units internally to ensure numerical stability:
- $G = 1.0$
- Distance = $1 \text{ AU}$
- Mass = $1 \ M_\odot$ (Solar Mass)
- Time unit = $\frac{1 \text{ year}}{2\pi}$

### The "Butterfly" Setup
We run two systems simultaneously:
- **System A (Baseline):** A carefully chosen N-body configuration (e.g., the chaotic Pythagorean three-body problem).
- **System B (Perturbed):** Exact same initial conditions, but a single body is given a microscopic velocity perturbation (e.g., $\Delta v_x = 10^{-5}$ computational units).

We calculate the Euclidean separation (divergence) over time:
$$ D(t) = \sqrt{\sum_i |\vec{r}_i^A - \vec{r}_i^B|^2} $$

Because the underlying system is chaotic, $D(t)$ starts near zero but eventually explodes exponentially, demonstrating the Butterfly Effect.

### Conservation Diagnostics
To ensure the divergence is due to actual physics and not numerical error, we strictly monitor invariants:
- Total Energy ($E = K + U$)
- Linear Momentum
- Angular Momentum
The on-screen HUD displays the relative energy drift in real-time.

## Visualization Features

The application is built to feel like an interactive astrophysics laboratory:
- **Cinematic Experiment Mode:** Automatically sequences through Initial Conditions, Release, Divergence, and Outcome phases to narratively demonstrate the butterfly effect.
- **Auto-Framing Camera:** The camera dynamically zooms and pans to keep both systems optimally framed as they expand or contract.
- **Live Divergence Graph:** An on-canvas real-time graph plots $\log_{10} D(t)$ to visually quantify the exact moment chaos overtakes the perturbed system.
- **High-Quality Trails:** Continuous line segments trace the orbital history, fading over time.
- **Dual View vs Overlay:** Switch between viewing the systems side-by-side or overlaid in the same coordinate space.

## Installation

This project is built using Python 3.12, the `uv` package manager, and `taichi` for high-performance numerical kernels.

```bash
# Clone the repository
# (Assuming you are in the project root)

# Install dependencies and setup environment
uv sync

# Run the interactive simulation
uv run cosmic-butterfly
```

## Running the Simulation

```bash
# Run the default (chaotic Pythagorean 3-body) scenario
uv run cosmic-butterfly --scenario pythagorean

# Run a stable Figure-8 orbit for comparison
uv run cosmic-butterfly --scenario figure8

# Enable GPU backend (if available) for better performance
uv run cosmic-butterfly --gpu
```

## Testing

The physics engine is validated against analytical two-body orbits and invariant conservation laws.
To run the automated validation tests:

```bash
uv run pytest
```

## Known Limitations
- The integration uses a fixed time step. Adaptive time-stepping (like RK45) would be better for extremely close encounters, but standard adaptive integrators break symplecticity. To maintain energy conservation, the Pythagorean scenario uses a very fine fixed time step.
- The visualization is 2D and rendering handles overlap simply by draw order.

## Technical Architecture
- **`physics/`**: Pure Taichi mathematical implementation of the N-body problem, units, and constants. 
- **`experiments/`**: Management of the initial scenarios and tracking divergence metrics between two simulation states.
- **`visualization/`**: Taichi GGUI implementation mapping the physics state arrays to visual elements.
- **`tests/`**: Automated scientific validation tests ensuring the math is correct.
