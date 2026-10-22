"""Copy the tutorial's 25 contrast-enhanced T1 scans without modifying originals."""
import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Extracted BraTS-PEDs download folder")
parser.add_argument("destination", type=Path, help="New BraTS_PEDs_T1c_N25 input folder")
args = parser.parse_args()
if args.destination.exists():
    parser.error("Destination already exists; choose a new folder to avoid overwriting data.")
ids = (Path(__file__).parent / "selected-participants.txt").read_text().splitlines()
selected = []
for subject in ids:
    matches = list(args.source.rglob(subject + "-t1c.nii.gz"))
    if len(matches) != 1:
        parser.error(f"Expected one T1c scan for {subject}; found {len(matches)}. No files copied.")
    selected.append((subject, matches[0]))
args.destination.mkdir(parents=True)
for subject, source in selected:
    folder = args.destination / subject
    folder.mkdir()
    shutil.copy2(source, folder / source.name)
    (folder / (subject + "-t1c.json")).write_text(json.dumps({"PatientID": subject}) + "\n")
print(f"Prepared {len(selected)} T1c scans in {args.destination.resolve()}")
