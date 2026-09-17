import taichi as ti
from .gravity import compute_all_accelerations

@ti.kernel
def velocity_verlet_step_1(
    n: int,
    dt: float,
    pos: ti.template(),
    vel: ti.template(),
    acc: ti.template()
):
    """
    First step of Velocity Verlet:
    r(t + dt) = r(t) + v(t)*dt + 0.5 * a(t) * dt^2
    """
    for i in range(n):
        pos[i] += vel[i] * dt + 0.5 * acc[i] * (dt ** 2)

@ti.kernel
def velocity_verlet_step_2(
    n: int,
    dt: float,
    vel: ti.template(),
    acc: ti.template(),
    acc_new: ti.template()
):
    """
    Second step of Velocity Verlet:
    v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt
    a(t) = a(t + dt)
    """
    for i in range(n):
        vel[i] += 0.5 * (acc[i] + acc_new[i]) * dt
        acc[i] = acc_new[i]

def step_velocity_verlet(
    n: int,
    dt: float,
    pos: ti.template(),
    vel: ti.template(),
    mass: ti.template(),
    acc: ti.template(),
    acc_new: ti.template()
):
    """
    Execute one full Velocity Verlet integration step.
    """
    # 1. Kick (position) using current v and a
    velocity_verlet_step_1(n, dt, pos, vel, acc)
    
    # 2. Compute new accelerations based on new positions
    compute_all_accelerations(n, pos, mass, acc_new)
    
    # 3. Drift (velocity) and update a = a_new
    velocity_verlet_step_2(n, dt, vel, acc, acc_new)
