import json
from pathlib import Path

from PIL import Image

from tinyocr.session import load_session, load_spec, predict_image

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_samples_match_their_labels():
    model = ROOT / "artifacts" / "recognizer.onnx"
    manifest = json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))
    spec = load_spec(model)
    session = load_session(model, threads=1)
    assert spec["height"] == 32
    assert spec["width"] == 160
    for row in manifest:
        with Image.open(ROOT / row["file"]) as image:
            assert predict_image(session, spec, image) == row["text"]
