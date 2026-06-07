"""
aqi_utils.py — EPA-style AQI sub-index calculation.

Converts forecast pollutant concentrations (μg/m³) into sub-indices via EPA
breakpoint interpolation; the reported AQI is the maximum sub-index (dominant
pollutant). Hourly forecast values approximate regulatory averaging windows
(PM 24 h, O₃ 8 h, NO₂ 1 h) for educational forecasting — not regulatory AQI.

AQI = ((I_high - I_low) / (C_high - C_low)) × (C - C_low) + I_low
"""

from __future__ import annotations

from typing import Any

# ── EPA breakpoint tables (concentration in μg/m³) ────────────────────────────
# Each entry: (C_low, C_high, I_low, I_high)

_PM25_BREAKPOINTS = [
    (0.0,   9.0,   0,   50),
    (9.1,   35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

_PM10_BREAKPOINTS = [
    (0,   54,  0,   50),
    (55,  154, 51,  100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

# O₃ breakpoints in μg/m³ (1 ppm ≈ 1960 μg/m³ at 25 °C); hourly values stand in for 8 h average
_O3_BREAKPOINTS = [
    (0,   108, 0,   50),
    (109, 140, 51,  100),
    (141, 170, 101, 150),
    (171, 210, 151, 200),
    (211, 400, 201, 300),
    (401, 800, 301, 500),
]

# NO₂ breakpoints in μg/m³ (1 ppb ≈ 1.88 μg/m³ at 25 °C); aligned with 1 h EPA standard
_NO2_BREAKPOINTS = [
    (0,    100,  0,   50),
    (101,  188,  51,  100),
    (189,  677,  101, 150),
    (678,  1220, 151, 200),
    (1221, 2349, 201, 300),
    (2350, 3853, 301, 500),
]

_BREAKPOINTS: dict[str, list[tuple]] = {
    "pm2_5": _PM25_BREAKPOINTS,
    "pm10":  _PM10_BREAKPOINTS,
    "o3":    _O3_BREAKPOINTS,
    "no2":   _NO2_BREAKPOINTS,
}

# ── EPA AQI categories (0-500 scale) ──────────────────────────────────────────
EPA_AQI_CATEGORIES: list[tuple[int, int, str, str]] = [
    # (low, high, label, hex_color)
    (0,   50,  "Good",                         "#00e400"),
    (51,  100, "Moderate",                     "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups","#ff7e00"),
    (151, 200, "Unhealthy",                    "#ff0000"),
    (201, 300, "Very Unhealthy",               "#8f3f97"),
    (301, 500, "Hazardous",                    "#7e0023"),
]


# ── Core interpolation ─────────────────────────────────────────────────────────

def _interpolate(c: float, breakpoints: list[tuple]) -> int | None:
    """
    Return the AQI sub-index for concentration `c` using the given breakpoint
    table.  Returns None if `c` is outside the defined range.
    """
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return round(aqi)
    return None


def calculate_sub_aqi(pollutant: str, concentration: float) -> int | None:
    """
    Calculate the AQI sub-index for a single pollutant.

    Parameters
    ----------
    pollutant : str
        One of 'pm2_5', 'pm10', 'o3', 'no2'.
    concentration : float
        Pollutant concentration in μg/m³.  Must be >= 0.

    Returns
    -------
    int or None
        AQI sub-index, or None if the pollutant is unknown or concentration
        is outside the defined breakpoint range.
    """
    if pollutant not in _BREAKPOINTS:
        return None
    if concentration < 0:
        concentration = 0.0
    return _interpolate(concentration, _BREAKPOINTS[pollutant])


def calculate_final_aqi(predicted_pollutants: dict[str, float]) -> dict[str, Any]:
    """
    Calculate the final AQI from a dict of predicted pollutant concentrations.

    The final AQI is the maximum sub-index across all available pollutants
    (the dominant-pollutant convention used by EPA and most AQI systems).

    Parameters
    ----------
    predicted_pollutants : dict[str, float]
        Mapping of pollutant name → concentration (μg/m³).
        Unknown or missing pollutants are silently skipped.

    Returns
    -------
    dict with keys:
        aqi               (int)  — final AQI (0-500)
        category          (str)  — human-readable category
        dominant_pollutant(str)  — pollutant driving the final AQI
        color             (str)  — hex color for the category
        sub_indices       (dict) — sub-index per pollutant
    """
    sub_indices: dict[str, int] = {}
    for pollutant, concentration in predicted_pollutants.items():
        val = calculate_sub_aqi(pollutant, float(concentration))
        if val is not None:
            sub_indices[pollutant] = val

    if not sub_indices:
        return {
            "aqi": None,
            "category": "Unknown",
            "dominant_pollutant": None,
            "color": "#999999",
            "sub_indices": {},
        }

    dominant = max(sub_indices, key=sub_indices.__getitem__)
    final_aqi = sub_indices[dominant]
    cat, color = aqi_category(final_aqi)

    return {
        "aqi": final_aqi,
        "category": cat,
        "dominant_pollutant": dominant,
        "color": color,
        "sub_indices": sub_indices,
    }


def aqi_category(aqi: int | float) -> tuple[str, str]:
    """
    Return (category_label, hex_color) for an EPA AQI value (0-500).

    Parameters
    ----------
    aqi : int or float
        AQI value on the 0-500 scale.

    Returns
    -------
    (label, color) tuple
    """
    for low, high, label, color in EPA_AQI_CATEGORIES:
        if low <= aqi <= high:
            return label, color
    if aqi > 500:
        return "Hazardous", "#7e0023"
    return "Unknown", "#999999"


def aqi_color(aqi: int | float) -> str:
    """Return hex color for an EPA AQI value (0-500 scale)."""
    return aqi_category(aqi)[1]


def aqi_label(aqi: int | float) -> str:
    """Return category label for an EPA AQI value (0-500 scale)."""
    return aqi_category(aqi)[0]
