#!/usr/bin/env python3
"""
Check that pinned package versions agree across the three files that declare
them by hand:

    env/singularity/thor.recipe        the Apptainer build (source of truth)
    env/docker/Dockerfile              the image published to ghcr.io
    env/singularity/requirements.txt   the readable record / offline wheelhouse

They are maintained separately, so they can silently drift. Publishing an image
built from different versions than the recipe a reviewer builds locally would
be worse than publishing nothing, so CI fails on any disagreement.

Run from the repository root:

    python3 .github/scripts/check_pins.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "env/singularity/thor.recipe"
DOCKERFILE = ROOT / "env/docker/Dockerfile"
REQUIREMENTS = ROOT / "env/singularity/requirements.txt"

PIN = re.compile(r"\b([A-Za-z0-9_.\-]+)==([0-9][0-9A-Za-z.\-]*)")

# Pinned per-architecture inside a shell conditional, so the same package
# legitimately appears at two versions. Checked separately below.
ARCH_DEPENDENT = {"torch", "torchvision", "torchaudio"}


def pins(path: Path) -> dict[str, set[str]]:
    """All name==version pairs on non-comment lines."""
    found: dict[str, set[str]] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for name, version in PIN.findall(stripped):
            found.setdefault(name.lower().replace("_", "-"), set()).add(version)
    return found


def main() -> int:
    for path in (RECIPE, DOCKERFILE, REQUIREMENTS):
        if not path.exists():
            print(f"missing file: {path.relative_to(ROOT)}")
            return 1

    recipe = pins(RECIPE)
    docker = pins(DOCKERFILE)
    reqs = pins(REQUIREMENTS)

    problems: list[str] = []

    # 1. The recipe and the Dockerfile must install identical versions: they
    #    are meant to produce the same image.
    for name in sorted(set(recipe) | set(docker)):
        in_recipe, in_docker = recipe.get(name), docker.get(name)
        if in_recipe != in_docker:
            problems.append(
                f"  {name}: thor.recipe={sorted(in_recipe) if in_recipe else 'absent'}"
                f"  Dockerfile={sorted(in_docker) if in_docker else 'absent'}"
            )

    # 2. requirements.txt records the x86_64 set and deliberately leaves the
    #    architecture-dependent packages unpinned, so only compare the rest.
    for name in sorted(set(recipe) - ARCH_DEPENDENT):
        if name in reqs and reqs[name] != recipe[name]:
            problems.append(
                f"  {name}: thor.recipe={sorted(recipe[name])}"
                f"  requirements.txt={sorted(reqs[name])}"
            )

    if problems:
        print("Pinned versions disagree between build files:\n")
        print("\n".join(problems))
        print(
            "\nUpdate all three so they match:\n"
            "  env/singularity/thor.recipe\n"
            "  env/docker/Dockerfile\n"
            "  env/singularity/requirements.txt"
        )
        return 1

    shared = sorted(set(recipe) & set(docker))
    print(f"Pins agree across all build files ({len(shared)} packages checked).")
    for name in shared:
        print(f"  {name}=={','.join(sorted(recipe[name]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
