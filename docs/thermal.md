# Experimental long-wave infrared view

Crosswalk Lab synthesizes a white-hot view for the drive monitor. A nearby pedestrian usually appears brighter than the road because the model assigns the pedestrian class a warmer surface. At greater distance the atmosphere reduces that contrast. These are assumed scene temperatures. The view has no camera calibration and supplies no temperature ground truth. It is a Crosswalk Lab experiment, independent of Waymo's sensor suite.

## Inputs

Each pixel takes its object class from CARLA's raw semantic image and its metric depth from a second camera at the same position, orientation, field of view, resolution and simulation tick. The render uses depth as an approximation of atmospheric path length. CARLA encodes the class in the red byte and depth in three bytes, with red least significant. The thermal computation never samples RGB brightness. [CARLA sensor reference](https://github.com/carla-simulator/carla/blob/1.0/Docs/ref_sensors.md) · [CARLA 1.0 class IDs](https://github.com/carla-simulator/carla/blob/1.0/LibCarla/source/carla/rpc/ObjectLabel.h)

## Radiance calculation

Let `B(T)` be the blackbody spectral radiance integrated from 8 to 14 µm. The implementation evaluates Planck's law at 0.1 µm intervals with uniform response across the band and trapezoidal integration. It precomputes the integrated surface terms for each class. Temperatures enter Planck's law in kelvin and the result has units of W/(m² sr). [NOAA's Planck function](https://ncc.nesdis.noaa.gov/planck.html)

For a surface of emissivity `ε`, the received band radiance combines emitted, reflected and atmospheric contributions. The reflection term assumes an opaque, diffuse surface under uniform surroundings. [FLIR Report Studio manual, chapter 21](https://support.flir.com/docdownload/assets/web/27eh/en-us/)

```text
surface      = ε × B(surface temperature) + (1 − ε) × B(reflected temperature)
transmission = exp(−0.0015 × distance in meters)
received     = transmission × surface + (1 − transmission) × B(air temperature)
```

The experiment fixes air temperature at 18 °C and reflected apparent temperature at 15 °C. A uniform extinction coefficient of 0.0015/m gives approximately 97% transmission at 20 m and 86% at 100 m. Sky pixels receive a fixed apparent sky temperature of 7 °C at the observer. A missing or invalid depth on a surface yields air radiance.

Example assumptions include 31 °C and 0.97 emissivity for clothed pedestrians, 26 °C and 0.90 for vehicle bodies, 22 °C and 0.95 for asphalt, and 15 °C and 0.97 for vegetation. Unclassified surfaces use 18 °C and 0.92. A single class estimate covers the entire visible object. Clothing, engine heat, individual materials, weather and solar loading can change or reverse real thermal contrast. The complete assumptions live in [the renderer](../crosswalk/thermal.py).

## Display

The same blackbody radiance endpoints, 5 °C for black and 35 °C for white, apply to every frame. Intermediate grayscale values are linear in received radiance. Seeded display dither adds up to one gray level before rounding. Two renders with the same inputs and seed are identical. The view illustrates visibility under these assumptions. Evaluating a real thermal camera would require an instrument response, a calibrated scene and measured comparison data.
