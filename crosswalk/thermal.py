"""Illustrative 8–14 µm view from aligned CARLA labels and depth."""
from __future__ import annotations

import numpy as np


def _band_radiance(celsius):
    """Blackbody radiance in W/(m² sr), assuming uniform spectral response."""
    wavelength = np.linspace(8e-6, 14e-6, 61)
    kelvin = np.asarray(celsius, dtype=np.float64)[..., None] + 273.15
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    spectral = (2 * h * c**2 / wavelength**5) / np.expm1(h * c / (wavelength * k * kelvin))
    return np.trapezoid(spectral, wavelength, axis=-1)


# CARLA 0.9.16 CityObjectLabel IDs. Each group shares an assumed surface
# temperature (°C) and band emissivity. Unknown tags use a generic surface.
_SURFACES = (
    ((1, 24),        22.0, 0.95),  # roads, road lines
    ((2, 3, 4, 26),  20.0, 0.93),  # sidewalks, buildings, walls, bridges
    ((5, 6, 8, 28),  18.0, 0.70),  # fences, poles, signs, guardrails
    ((9,),           15.0, 0.97),  # vegetation
    ((10, 25),       17.0, 0.94),  # terrain, ground
    ((12, 13),       31.0, 0.97),  # clothed pedestrians and riders
    ((14, 15, 16, 17, 18), 26.0, 0.90),  # motor vehicles
    ((19, 27),       18.0, 0.75),  # bicycles, railway tracks
    ((23,),          16.0, 0.98),  # water
)
_SKY = 11
_SKY_RADIANCE = np.float32(_band_radiance(7.0))
_AIR_RADIANCE = np.float32(_band_radiance(18.0))
_EXTINCTION_PER_M = np.float32(0.0015)
_DISPLAY_MIN, _DISPLAY_MAX = _band_radiance([5.0, 35.0]).astype(np.float32)


def _surface_radiances():
    temperature = np.full(256, 18.0)
    emissivity = np.full(256, 0.92)
    for tags, celsius, epsilon in _SURFACES:
        temperature[list(tags)] = celsius
        emissivity[list(tags)] = epsilon
    reflected = _band_radiance(15.0)
    return (emissivity * _band_radiance(temperature) + (1 - emissivity) * reflected).astype(np.float32)


_SURFACE_RADIANCE = _surface_radiances()


def thermal_rgb(tags, depth_m, seed=0):
    """Render a white-hot RGB image. Both inputs must describe the same frame and rays."""
    tags = np.asarray(tags)
    depth = np.asarray(depth_m, dtype=np.float32)
    if tags.ndim != 2 or tags.shape != depth.shape:
        raise ValueError("Expected aligned H×W semantic tags and metric depth")
    if tags.dtype != np.uint8:
        raise ValueError("Expected uint8 CARLA semantic IDs")

    # Invalid distances have no modeled surface contribution. Sky already
    # represents downwelling radiance at the observer and bypasses this path.
    distance = np.where(np.isfinite(depth) & (depth >= 0), depth, np.float32(np.inf))
    transmission = np.exp(-_EXTINCTION_PER_M * distance)
    received = _AIR_RADIANCE + transmission * (_SURFACE_RADIANCE[tags] - _AIR_RADIANCE)
    received[tags == _SKY] = _SKY_RADIANCE

    level = (received - _DISPLAY_MIN) * (255 / (_DISPLAY_MAX - _DISPLAY_MIN))
    # A seed per frame gives reproducible display dither of at most one gray level.
    level += 2 * np.random.default_rng(seed).random(tags.shape, dtype=np.float32) - 1
    gray = np.clip(np.rint(level), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)
