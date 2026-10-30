"""Resize a text line to the fixed tensor the ONNX model consumes."""

import numpy as np
from PIL import Image


def prepare_line(image: Image.Image, height: int, width: int) -> np.ndarray:
    """Return float32 array shaped (1, height, width), values in [0, 1].

    The line is scaled to `height` and padded on the right. A dark page
    (light glyphs) is inverted so the network always sees dark text.
    """
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    gray = image.convert("L")
    src_w, src_h = gray.size
    if src_w < 1 or src_h < 1:
        raise ValueError("empty image")
    new_w = max(1, int(round(src_w * (height / src_h))))
    if new_w > width:
        resized = gray.resize((width, height), Image.Resampling.BILINEAR)
        canvas = resized
    else:
        resized = gray.resize((new_w, height), Image.Resampling.BILINEAR)
        bg = int(resized.getpixel((0, 0)))
        canvas = Image.new("L", (width, height), color=bg)
        canvas.paste(resized, (0, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if float(arr.mean()) < 0.5:
        arr = 1.0 - arr
    return arr.reshape(1, height, width)
