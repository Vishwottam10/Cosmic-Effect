import math

# We choose a computational unit system where:
# - Distance = 1 Astronomical Unit (AU)
# - Mass = 1 Solar Mass (M_sun)
# - G = 1.0
# 
# In physical units, G is approx 4*pi^2 (AU^3 / M_sun / yr^2).
# Since our computational G = 1.0, our computational time unit T_comp relates to years by:
# T_comp = 1 yr / (2 * pi)

COMPUTATIONAL_G = 1.0
PHYSICAL_G = 4.0 * math.pi**2

def time_to_years(t_comp: float) -> float:
    """Convert computational time units to physical years."""
    return t_comp / (2.0 * math.pi)

def years_to_time(years: float) -> float:
    """Convert physical years to computational time units."""
    return years * 2.0 * math.pi

def distance_to_au(d_comp: float) -> float:
    """Convert computational distance to AU."""
    return d_comp

def mass_to_solar(m_comp: float) -> float:
    """Convert computational mass to Solar Masses."""
    return m_comp
