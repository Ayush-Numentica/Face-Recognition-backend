"""
utils.py
--------
Utility functions for the Face Recognition backend.
Handles image encoding/decoding and file operations.
"""

import base64
import os
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str):
    """
    Decode a base64-encoded image string into a BGR numpy array.

    Args:
        base64_string: Base64 encoded image (with or without data-URL prefix)

    Returns:
        BGR numpy array, or None if decoding fails
    """
    try:
        # Strip data-URL prefix (e.g. "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"decode_base64_image error: {e}")
        return None


def encode_image_to_base64(image) -> str:
    """
    Encode a BGR numpy array to a base64 JPEG string.

    Args:
        image: BGR numpy array

    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def save_image(image, path: str) -> bool:
    """
    Save a BGR numpy array to disk.

    Args:
        image: BGR numpy array
        path:  Absolute file path

    Returns:
        True on success, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        logger.error(f"save_image error: {e}")
        return False


def resize_keep_aspect(image, max_dim: int = 640):
    """
    Resize an image so its longest side is at most *max_dim*,
    preserving the original aspect ratio.

    Args:
        image:   BGR numpy array
        max_dim: Maximum allowed dimension in pixels

    Returns:
        Resized numpy array (or original if already small enough)
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    if h >= w:
        scale = max_dim / h
    else:
        scale = max_dim / w

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
