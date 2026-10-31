"""Package and validate an audio submission without running any scoring models."""

import argparse
import tempfile
import zipfile
from pathlib import Path

from .contract import SENTENCES, Manifest, validate_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    Manifest.model_validate_json(args.manifest.read_text())
    expected = {sid + ".wav" for sid in SENTENCES}
    actual = {p.name for p in args.audio.iterdir()}
    if actual != expected:
        parser.error(
            f"Audio directory must contain exactly the {len(expected)} required WAV files."
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "submission.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(args.manifest, "submission.json")
            for name in sorted(expected):
                z.write(args.audio / name, "audio/" + name)
        _, result = validate_archive(archive, tmp / "validated")
        # Refuse to overwrite an existing package.
        with args.output.open("xb") as target, archive.open("rb") as source:
            import shutil

            shutil.copyfileobj(source, target)
        print(
            f"Validated {result['files']} files, {result['seconds']:.1f}s. Saved {args.output}."
        )


if __name__ == "__main__":
    main()
