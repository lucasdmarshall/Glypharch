"""Find installed font files by filename, without hard-coding one distro."""

import os
from functools import lru_cache
from pathlib import Path

TRAIN_FILENAMES = (
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold.ttf",
    "DejaVuSerif.ttf",
    "LiberationSans-Regular.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Italic.ttf",
    "LiberationMono-Regular.ttf",
    "LiberationMono-Bold.ttf",
    "LiberationSerif-Regular.ttf",
    "NotoSans-Regular.ttf",
    "NotoSans-Bold.ttf",
    "NotoSans-Italic.ttf",
    "Arimo-Regular.ttf",
    "Arimo-Bold.ttf",
    "Cousine-Regular.ttf",
    "Tinos-Regular.ttf",
    "Inter-Regular.ttf",
    "Inter-Bold.ttf",
    "SourceSans3-Regular.ttf",
    "PublicSans-Regular.ttf",
)

HELDOUT_FILENAMES = (
    "Cantarell-Regular.otf",
    "Cantarell-Bold.otf",
    "JetBrainsMono-Regular.ttf",
    "CascadiaMono.ttf",
)

_FONT_ROOTS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
    Path.home() / "Library/Fonts",
)


@lru_cache(maxsize=1)
def font_index() -> dict[str, str]:
    found: dict[str, str] = {}
    wanted = set(TRAIN_FILENAMES) | set(HELDOUT_FILENAMES)
    for root in _FONT_ROOTS:
        if not root.is_dir():
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name in wanted and name not in found:
                    found[name] = str(Path(dirpath) / name)
    return found


def resolve_fonts(filenames: tuple[str, ...], overrides: list[str] | None = None) -> list[str]:
    if overrides:
        missing = [path for path in overrides if not Path(path).is_file()]
        if missing:
            raise SystemExit(f"font file not found: {', '.join(missing)}")
        return list(overrides)
    index = font_index()
    return [index[name] for name in filenames if name in index]
