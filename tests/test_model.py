import pytest

torch = pytest.importorskip("torch")

from tinyocr.config import CHARSET, DETAIL_H, DETAIL_TIME_STEPS, DETAIL_W, IMG_H, IMG_W, TIME_STEPS
from tinyocr.model import DetailRecognizer, TinyRecognizer, recognizer_for


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


def test_detail_loads_tiny_weights_and_keeps_forty_steps():
    nclass = len(CHARSET) + 1
    tiny = TinyRecognizer(nclass)
    detail = recognizer_for(DETAIL_H, nclass)
    assert isinstance(detail, DetailRecognizer)
    assert isinstance(recognizer_for(IMG_H, nclass), TinyRecognizer)
    detail.load_state_dict(tiny.state_dict())
    assert sum(p.numel() for p in detail.parameters()) == sum(p.numel() for p in tiny.parameters())
    detail.eval()
    images = torch.zeros(2, 1, DETAIL_H, DETAIL_W)
    with torch.no_grad():
        log_probs = detail(images)
    assert detail.cnn(images).shape[2:] == (1, DETAIL_TIME_STEPS)
    assert log_probs.shape == (2, DETAIL_TIME_STEPS, nclass)
    probs = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)
