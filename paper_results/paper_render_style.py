"""Backend-independent materials and studio conventions for paper renderings.

Colors are sRGB hex values. Convert to linear RGB only for renderers that
expect linear material inputs (such as Newton's RTX viewer).
"""

from dataclasses import asdict, dataclass


def rgb(hex_color: str, *, linear: bool = False) -> tuple[float, float, float]:
    """Convert an sRGB hex color to normalized sRGB or linear RGB."""
    value = hex_color.removeprefix("#")
    if len(value) != 6:
        raise ValueError("Expected a six-digit RGB color")
    channels = tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    if linear:
        return tuple(
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in channels
        )
    return channels


@dataclass(frozen=True)
class PaperStudio:
    """Shared visual intent; backend light intensities require calibration."""

    version: str = "paper-studio-v1"
    robot: str = "#F1552E"
    mount: str = "#33363B"
    target: str = "#0EAD69"
    obstacle: str = "#7B2CBF"
    manipulated_object: str = "#E7AC42"
    surface: str = "#DEE1E5"
    backdrop: str = "#C9CDD2"
    robot_roughness: float = 0.72
    backdrop_start: float = 3.0
    backdrop_radius: tuple[float, float] = (2.0, 2.0)
    key_rotation_degrees: tuple[float, float, float] = (29.0, 14.0, 0.0)
    key_angle_degrees: float = 28.0
    fill_rotation_degrees: tuple[float, float, float] = (-30.0, -25.0, 0.0)
    rim_rotation_degrees: tuple[float, float, float] = (40.0, 15.0, 180.0)
    fill_angle_degrees: float = 50.0
    # Newton RTX calibration, kept explicit instead of assuming other backends
    # use the same physical units for their light intensities.
    rtx_ambient: float = 60.0
    rtx_key: float = 1000.0
    rtx_fill: float = 160.0
    rtx_rim: float = 160.0
    rtx_backdrop_emission: float = 0.08

    def manifest(self) -> dict:
        """Return serializable settings for figure provenance."""
        return asdict(self)


PAPER_STUDIO = PaperStudio()
