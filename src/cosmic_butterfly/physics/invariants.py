import taichi as ti
from .constants import G, FLOAT_TYPE, vec2

@ti.kernel
def compute_kinetic_energy(n: int, vel: ti.template(), mass: ti.template()) -> FLOAT_TYPE:
    k = ti.cast(0.0, FLOAT_TYPE)
    for i in range(n):
        k += 0.5 * mass[i] * vel[i].norm_sqr()
    return k

@ti.kernel
def compute_potential_energy(n: int, pos: ti.template(), mass: ti.template()) -> FLOAT_TYPE:
    u = ti.cast(0.0, FLOAT_TYPE)
    for i in range(n):
        for j in range(i + 1, n):
            r = (pos[j] - pos[i]).norm()
            u -= G * mass[i] * mass[j] / r
    return u

def compute_total_energy(n: int, pos: ti.template(), vel: ti.template(), mass: ti.template()) -> float:
    return compute_kinetic_energy(n, vel, mass) + compute_potential_energy(n, pos, mass)

@ti.kernel
def compute_linear_momentum(n: int, vel: ti.template(), mass: ti.template()) -> vec2:
    p = vec2(0.0, 0.0)
    for i in range(n):
        p += mass[i] * vel[i]
    return p

@ti.kernel
def compute_angular_momentum(n: int, pos: ti.template(), vel: ti.template(), mass: ti.template()) -> FLOAT_TYPE:
    L = ti.cast(0.0, FLOAT_TYPE)
    for i in range(n):
        # 2D cross product: x * vy - y * vx
        r = pos[i]
        v = vel[i]
        L += mass[i] * (r[0] * v[1] - r[1] * v[0])
    return L
