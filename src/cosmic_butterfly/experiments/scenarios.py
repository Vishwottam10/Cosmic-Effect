import numpy as np
from dataclasses import dataclass

@dataclass
class Scenario:
    name: str
    num_bodies: int
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    dt: float = 0.001
    
    def copy(self):
        return Scenario(
            name=self.name,
            num_bodies=self.num_bodies,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            mass=self.mass.copy(),
            dt=self.dt
        )

def create_pythagorean_three_body() -> Scenario:
    """
    A classic chaotic 3-body problem known as the Pythagorean three-body problem.
    Initial positions form a 3-4-5 right triangle with masses 3, 4, and 5.
    Initial velocities are zero.
    This system is highly chaotic and exhibits a lot of close encounters.
    """
    pos = np.array([
        [1.0, 3.0],   # Mass 3
        [-2.0, -1.0], # Mass 4
        [1.0, -1.0]   # Mass 5
    ], dtype=np.float64)
    
    vel = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ], dtype=np.float64)
    
    mass = np.array([3.0, 4.0, 5.0], dtype=np.float64)
    
    return Scenario("Pythagorean 3-Body", 3, pos, vel, mass, dt=0.000025)

def create_hierarchical_three_body() -> Scenario:
    """
    A star with a planet, and a moon orbiting the planet.
    This can be stable or unstable depending on distances.
    """
    # Star
    m1 = 1.0
    r1 = np.array([0.0, 0.0])
    v1 = np.array([0.0, 0.0])
    
    # Planet
    m2 = 0.01
    d2 = 1.0
    v2_mag = np.sqrt(1.0 * m1 / d2)
    r2 = np.array([d2, 0.0])
    v2 = np.array([0.0, v2_mag])
    
    # Moon
    m3 = 0.0001
    d3 = 0.05
    v3_mag = np.sqrt(1.0 * m2 / d3)
    r3 = r2 + np.array([d3, 0.0])
    v3 = v2 + np.array([0.0, v3_mag])
    
    pos = np.vstack([r1, r2, r3]).astype(np.float64)
    vel = np.vstack([v1, v2, v3]).astype(np.float64)
    mass = np.array([m1, m2, m3], dtype=np.float64)
    
    return Scenario("Hierarchical 3-Body", 3, pos, vel, mass, dt=0.001)

def create_figure_eight() -> Scenario:
    """
    Stable figure-eight orbit of 3 equal masses.
    """
    m = 1.0
    pos = np.array([
        [0.97000436, -0.24308753],
        [-0.97000436, 0.24308753],
        [0.0, 0.0]
    ], dtype=np.float64)
    
    vel = np.array([
        [0.4662036850, 0.4323657300],
        [0.4662036850, 0.4323657300],
        [-2.0 * 0.4662036850, -2.0 * 0.4323657300]
    ], dtype=np.float64)
    
    mass = np.array([m, m, m], dtype=np.float64)
    
    return Scenario("Figure-8", 3, pos, vel, mass, dt=0.001)
