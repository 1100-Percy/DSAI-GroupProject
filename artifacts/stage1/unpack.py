"""Reassemble and verify the complete bundle using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "stage1_download")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    manifest = json.loads((source / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage1_unpack_") as temporary:
        merged = Path(temporary) / "stage1_complete.zip"
        with merged.open("wb") as target:
            for part in manifest["parts"]:
                path = source / part["filename"]
                assert path.stat().st_size == part["bytes"]
                assert sha256(path) == part["sha256"], f"Damaged part: {path.name}"
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        target.write(chunk)
        assert sha256(merged) == manifest["complete_zip"]["sha256"]
        with zipfile.ZipFile(merged) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                destination = (output / name).resolve()
                assert destination.is_relative_to(output), "Unsafe archive path"
            archive.extractall(output)
        contents = json.loads((output / "BUNDLE_CONTENTS.json").read_text(encoding="utf-8"))
        for name, expected in contents.items():
            assert sha256(output / name) == expected, f"Extracted hash mismatch: {name}"
    print(f"Verified and extracted {len(contents)} files into {output}")


if __name__ == "__main__":
    main()
