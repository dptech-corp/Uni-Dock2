#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import shlex
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_PATTERN = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<specifier>\s*[<>=!~].*)?\s*$")


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def display_path(path: Path) -> Path:
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT)
    return path


def to_conda(requirement: str, overrides: dict[str, str]) -> str:
    if requirement in overrides:
        return overrides[requirement]

    match = REQUIREMENT_PATTERN.fullmatch(requirement)
    if match is None:
        raise ValueError(f"Unsupported dependency {requirement!r}; add a conda override for non-standard requirements")

    name = match.group("name")
    normalized_name = normalize_package_name(name)
    if normalized_name in overrides:
        return overrides[normalized_name]

    specifier = (match.group("specifier") or "").strip()
    if specifier:
        return f"{normalized_name} {specifier}"
    return normalized_name


def replace_block(
    path: Path,
    block_name: str,
    content: str,
    comment: str,
    check: bool,
) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start_marker = f"{comment} BEGIN GENERATED: {block_name}"
    end_marker = f"{comment} END GENERATED: {block_name}"

    start_matches = [index for index, line in enumerate(lines) if line.strip() == start_marker]
    end_matches = [index for index, line in enumerate(lines) if line.strip() == end_marker]
    if len(start_matches) != 1 or len(end_matches) != 1:
        raise ValueError(f"Expected exactly one {block_name!r} block in {display_path(path)}")

    start = start_matches[0]
    end = end_matches[0]
    if end <= start:
        raise ValueError(f"Invalid generated block order in {display_path(path)}")

    marker_line = lines[start]
    indent = marker_line[: len(marker_line) - len(marker_line.lstrip())]
    replacement_lines = [f"{indent}{start_marker}\n"]
    replacement_lines.extend(f"{indent}{line}\n" if line else "\n" for line in content.splitlines())
    replacement_lines.append(f"{indent}{end_marker}\n")

    updated = "".join(lines[:start] + replacement_lines + lines[end + 1 :])
    changed = updated != text
    if changed and not check:
        path.write_text(updated, encoding="utf-8")
        print(f"Updated {display_path(path)} ({block_name})")
    return changed


def replace_pattern(
    path: Path,
    label: str,
    pattern: str,
    replacement: str,
    check: bool,
) -> bool:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {label!r} value in {display_path(path)}")

    changed = updated != text
    if changed and not check:
        path.write_text(updated, encoding="utf-8")
        print(f"Updated {display_path(path)} ({label})")
    return changed


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize dependency blocks from pyproject.toml")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    with (ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    dependency_config = config["tool"]["unidock2"]["deps"]
    overrides = dependency_config.get("conda-overrides", {})
    run_dependencies = [to_conda(requirement, overrides) for requirement in config["project"]["dependencies"]]
    run_dependencies.extend(dependency_config["conda-only-run"])
    build_tools = dependency_config["conda-build-tools"]
    channels = dependency_config["channels"]

    cmake_requirements = [requirement for requirement in build_tools if requirement.startswith("cmake ")]
    if len(cmake_requirements) != 1:
        raise ValueError("conda-build-tools must contain exactly one CMake requirement")
    cmake_match = re.fullmatch(r"cmake\s+>=\s*(?P<version>\S+)", cmake_requirements[0])
    if cmake_match is None:
        raise ValueError("The CMake build requirement must use the form 'cmake >=VERSION'")
    cmake_minimum = cmake_match.group("version")

    conda_build_block = "\n".join(f"- {item}" for item in build_tools)
    conda_run_block = "\n".join(f"- {item}" for item in run_dependencies)
    channel_args = shell_join(argument for channel in channels for argument in ("-c", channel))
    docker_dependencies = [
        *build_tools,
        *dependency_config["docker-extra"],
        *run_dependencies,
    ]
    docker_block = (
        f"RUN micromamba install -y {channel_args} \\\n"
        f"    {shell_join(docker_dependencies)} \\\n"
        "    && micromamba clean -a"
    )
    readme_block = f"conda install {shell_join(run_dependencies)} {channel_args} --no-repodata-use-zst"

    changed = [
        replace_pattern(
            ROOT / "engine/CMakeLists.txt",
            "CMake minimum version",
            r"^cmake_minimum_required\(VERSION [^)]+\)$",
            f"cmake_minimum_required(VERSION {cmake_minimum})",
            args.check,
        ),
        replace_pattern(
            ROOT / "README.md",
            "CMake prerequisite",
            r"^\* `CMake >= [^`]+`$",
            f"* `CMake >= {cmake_minimum}`",
            args.check,
        ),
        replace_block(
            ROOT / "conda-recipe/meta.yaml",
            "conda build tools",
            conda_build_block,
            "#",
            args.check,
        ),
        replace_block(
            ROOT / "conda-recipe/meta.yaml",
            "conda run dependencies",
            conda_run_block,
            "#",
            args.check,
        ),
        replace_block(
            ROOT / "docker/Dockerfile.base",
            "conda and Docker dependencies",
            docker_block,
            "#",
            args.check,
        ),
        replace_block(
            ROOT / "README.md",
            "conda run dependencies",
            readme_block,
            "#",
            args.check,
        ),
    ]

    if args.check and any(changed):
        print(
            "Dependency blocks are stale; run tools/sync_deps.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
