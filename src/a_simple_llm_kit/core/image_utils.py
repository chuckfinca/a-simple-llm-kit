"""
Image processing utilities for the LLM Server framework.
"""

import io
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS

from a_simple_llm_kit.core import logging


def extract_gps_from_image(image_bytes: bytes) -> dict[str, Any] | None:
    """
    Extract GPS coordinates from photo EXIF data if present.

    Args:
        image_bytes: Raw image data

    Returns:
        Dictionary with latitude, longitude, and source, or None if no GPS data
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif = image.getexif()

        if not exif:
            logging.debug("No EXIF data found in image")
            return None

        # Look for GPS info
        gps_info = None
        for tag, value in exif.items():
            if TAGS.get(tag) == "GPSInfo":
                gps_info = value
                break

        if not gps_info:
            logging.debug("No GPS data found in EXIF")
            return None

        logging.info(f"Found GPS data in EXIF: {gps_info}")

        # Parse GPS coordinates
        lat = _parse_gps_coord(gps_info.get(2), gps_info.get(1))
        lng = _parse_gps_coord(gps_info.get(4), gps_info.get(3))

        if lat is None or lng is None:
            logging.warning("Could not parse GPS coordinates from EXIF data")
            return None

        location_data = {"latitude": lat, "longitude": lng, "source": "photo_exif"}

        logging.info(f"Extracted photo location: {location_data}")
        return location_data

    except Exception as e:
        logging.warning(f"Error extracting GPS from EXIF: {e}")
        return None


def _parse_gps_coord(coord_tuple, ref) -> float | None:
    """Parse GPS coordinate from EXIF format to decimal degrees."""
    try:
        if not coord_tuple or not ref:
            return None

        degrees = float(coord_tuple[0])
        minutes = float(coord_tuple[1]) if len(coord_tuple) > 1 else 0.0
        seconds = float(coord_tuple[2]) if len(coord_tuple) > 2 else 0.0

        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

        if ref.upper() in ["S", "W"]:
            decimal = -decimal

        return decimal

    except (TypeError, ValueError, IndexError) as e:
        logging.warning(f"Error parsing GPS coordinate {coord_tuple}, {ref}: {e}")
        return None
