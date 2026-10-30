import pytest

torch = pytest.importorskip("torch")

from tinyocr.config import CHARSET, IMG_H, IMG_W, TIME_STEPS
from tinyocr.model import TinyRecognizer


def test_output_is_a_ctc_distribution():
    model = TinyRecognizer(len(CHARSET) + 1)
    model.eval()
    images = torch.zeros(2, 1, IMG_H, IMG_W)
    with torch.no_grad():
        log_probs = model(images)
    assert model.cnn(images).shape[2] == 1
    assert log_probs.shape == (2, TIME_STEPS, len(CHARSET) + 1)
    probs = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)
