from __future__ import annotations

from pathlib import Path
from typing import Any


def _summarize(name: str, obj: Any) -> None:
    """
    Print a compact summary for common converted R object types.
    `pyreadr` typically converts:
      - R data.frame -> pandas.DataFrame
      - R list -> dict-like / OrderedDict (depending on structure)
      - vectors/matrices -> numpy arrays / pandas Series (varies)
    """
    type_name = type(obj).__name__
    details: list[str] = []

    # Avoid importing heavy deps unless we need them.
    shape = getattr(obj, "shape", None)
    if shape is not None:
        details.append(f"shape={shape}")

    columns = getattr(obj, "columns", None)
    if columns is not None:
        try:
            details.append(f"cols={len(columns)}")
        except Exception:
            pass

    length = None
    try:
        length = len(obj)  # type: ignore[arg-type]
    except Exception:
        length = None
    if length is not None and shape is None:
        details.append(f"len={length}")

    tail = f" ({', '.join(details)})" if details else ""
    print(f"- {name}: {type_name}{tail}")


def load_rdata(path: str | Path) -> dict[str, Any]:
    """
    Loads an .RData/.rda file and returns a dict: {object_name: object}.

    Install deps:
      pip install pyreadr pandas
    """
    try:
        import pyreadr  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pyreadr\n"
            "Install with: pip install pyreadr pandas\n"
            "Then re-run this script."
        ) from exc

    result = pyreadr.read_r(str(path))
    # `result` behaves like an OrderedDict[str, Any]
    return dict(result)


def main() -> None:
    here = Path(__file__).resolve().parent

    files = {
        "BSL": here / "BSL_allChan.RData",
        "SENSORY": here / "SENSORY_allChan.RData",
        "DELAY": here / "DELAY_allChan.RData",
    }

    loaded: dict[str, dict[str, Any]] = {}
    for label, p in files.items():
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

        print(f"\n{label}: loading {p.name} ...")
        objs = load_rdata(p)
        print(f"{label}: objects={list(objs.keys())}")
        for name, obj in objs.items():
            _summarize(name, obj)
        loaded[label] = objs

    # Common convenience: if each .RData contains exactly one object, expose it directly.
    for label, objs in loaded.items():
        if len(objs) == 1:
            only_name, only_obj = next(iter(objs.items()))
            print(f"\n{label}: single object shortcut -> {label.lower()} = '{only_name}'")
            globals()[label.lower()] = only_obj  # for interactive runs


if __name__ == "__main__":
    main()