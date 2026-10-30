import numpy as np
from PIL import Image

from tinyocr.preprocess import prepare_line


def test_narrow_line_is_padded_to_the_model_width():
    image = Image.new("L", (40, 20), color=255)
    array = prepare_line(image, height=32, width=160)
    assert array.shape == (1, 32, 160)
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0
    assert float(array.max()) <= 1.0


def test_wide_line_is_squeezed_to_the_model_width():
    image = Image.new("L", (800, 40), color=255)
    array = prepare_line(image, height=32, width=160)
    assert array.shape == (1, 32, 160)


def test_dark_page_is_inverted_to_dark_text_on_light():
    image = Image.new("L", (30, 16), color=0)
    image.putpixel((4, 4), 255)
    array = prepare_line(image, height=32, width=160)
    assert float(array.mean()) > 0.5
