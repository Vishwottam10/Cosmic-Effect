import taichi as ti
from .constants import G, EPSILON, FLOAT_TYPE, vec2

@ti.func
def compute_acceleration_func(
    i: int, 
    n: int, 
    pos: ti.template(), 
    mass: ti.template()
) -> vec2:
    """
    Compute gravitational acceleration on body i from all other bodies.
    This is a Taichi device function.
    """
    acc = vec2(0.0, 0.0)
    for j in range(n):
        if i != j:
            r_ij = pos[j] - pos[i]
            dist_sq = r_ij.norm_sqr()
            # F = G * m1 * m2 / r^2
            # a = F / m1 = G * m2 / r^2
            # vector a = (G * m2 / r^3) * r_vec
            # To prevent singularity, we use dist_sq + EPSILON**2
            denom = (dist_sq + EPSILON**2) ** 1.5
            acc += G * mass[j] * r_ij / denom
    return acc

@ti.kernel
def compute_all_accelerations(
    n: int,
    pos: ti.template(),
    mass: ti.template(),
    acc: ti.template()
):
    """
    Compute accelerations for all bodies and store in acc field.
    """
    for i in range(n):
        acc[i] = compute_acceleration_func(i, n, pos, mass)
