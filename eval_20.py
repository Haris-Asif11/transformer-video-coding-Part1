from pathlib import Path
import sys
import os
import shutil
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for evaluation of (un-)compressed Cityscapes sequences for instance and semantic segmentation")
    parser.add_argument("-s", "--source", type=str, help="Path to sequences")
    parser.add_argument("-g", "--gtdir", type=str, default="/home/LMS/sequences/Autonomous_driving/Cityscapes/gtFine/val")
    parser.add_argument("-d", "--destination", type=str, help="Output path")
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    source_path = Path(args.source)
    gt_path = Path(args.gtdir)
    destination_path = Path(args.destination)
    for path in sorted(gt_path.rglob("*_gtFine_instanceIds.png")):
        city = path.stem.split("_")[0]
        os.makedirs(destination_path / city, exist_ok=True)
        rel = str(path.relative_to(gt_path)).replace("gtFine_instanceIds", "leftImg8bit")
        img = source_path / rel
        shutil.copy(img, destination_path / rel)

if __name__ == "__main__":
    main(sys.argv[1:])