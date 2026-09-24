"""MANIFEST.sha256: checksum de cada archivo de un dataset (formato compatible con `sha256sum -c`)."""
from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST = "MANIFEST.sha256"


class ManifestError(RuntimeError):
    pass


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(root: Path) -> Path:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST)
    lines = [f"{sha256(p)}  {p.relative_to(root).as_posix()}" for p in files]
    out = root / MANIFEST
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_manifest(root: Path) -> int:
    """Verifica todos los checksums; lanza ManifestError si falta, sobra o cambió algún archivo."""
    path = root / MANIFEST
    if not path.exists():
        raise ManifestError(f"No existe {path}")
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        expected[rel] = digest
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != MANIFEST}
    problems = [f"falta: {r}" for r in sorted(set(expected) - present)]
    problems += [f"no registrado: {r}" for r in sorted(present - set(expected))]
    problems += [f"checksum distinto: {r}" for r in sorted(set(expected) & present) if sha256(root / r) != expected[r]]
    if problems:
        raise ManifestError(f"El dataset {root.name} no coincide con su manifiesto:\n  " + "\n  ".join(problems))
    return len(expected)
