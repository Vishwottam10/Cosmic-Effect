import taichi as ti
import numpy as np
from .constants import FLOAT_TYPE
from .gravity import compute_all_accelerations
from .integrator import step_velocity_verlet

@ti.data_oriented
class Simulation:
    """
    Manages the state and evolution of an N-body system.
    """
    def __init__(self, num_bodies: int):
        self.n = num_bodies
        self.time = 0.0

        # Physical state
        self.pos = ti.Vector.field(2, dtype=FLOAT_TYPE, shape=self.n)
        self.vel = ti.Vector.field(2, dtype=FLOAT_TYPE, shape=self.n)
        self.mass = ti.field(dtype=FLOAT_TYPE, shape=self.n)
        
        # Integration intermediates
        self.acc = ti.Vector.field(2, dtype=FLOAT_TYPE, shape=self.n)
        self.acc_new = ti.Vector.field(2, dtype=FLOAT_TYPE, shape=self.n)

    def set_state(self, pos_np: np.ndarray, vel_np: np.ndarray, mass_np: np.ndarray):
        """Initialize the state from numpy arrays."""
        assert pos_np.shape == (self.n, 2)
        assert vel_np.shape == (self.n, 2)
        assert mass_np.shape == (self.n,)

        self.pos.from_numpy(pos_np)
        self.vel.from_numpy(vel_np)
        self.mass.from_numpy(mass_np)

        # Compute initial accelerations
        compute_all_accelerations(self.n, self.pos, self.mass, self.acc)
        self.time = 0.0

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a copy of the current state as numpy arrays."""
        return self.pos.to_numpy(), self.vel.to_numpy(), self.mass.to_numpy()

    def step(self, dt: float):
        """Advance the simulation by time dt."""
        step_velocity_verlet(
            self.n, dt, 
            self.pos, self.vel, self.mass, 
            self.acc, self.acc_new
        )
        self.time += dt
