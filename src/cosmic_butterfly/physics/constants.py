import taichi as ti

# Universal gravitational constant in computational units.
# We set G = 1.0 to keep numerical values stable.
G = 1.0

# Softening parameter to prevent singularities during close encounters
EPSILON = 1e-6

# Default precision for simulation state
FLOAT_TYPE = ti.f64
vec2 = ti.types.vector(2, FLOAT_TYPE)
