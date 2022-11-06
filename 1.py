import argparse
from pathlib import Path
import subprocess

import pandas as pd
from progressbar import ProgressBar


def get_frames(subdirectory, video, odir):
    """
    Run the ffmpeg command to extract the frames. If subdirectory is
    FinalTest than it will extract frames at default frame rate else it
    will extract frames at 0.7fps.

    Args:
        subdirectory: Specify it as either Train, Test, Validation or
            FinalTest.
        video: Path to input video.
        odir: Path to store output video.
    """
    if subdirectory == "FinalTest":
        subprocess.run(f"ffmpeg -i {video} "
                       f"{odir}/{video.parts[-1][:-4]}_%d.jpeg "
                       "-loglevel quiet", shell=True, check=True)
    else:
        subprocess.run(f"ffmpeg -i {video} -vf fps=0.7 "
                       f"{odir}/{video.parts[-1][:-4]}_%1d.jpeg "
                       "-loglevel quiet", shell=True, check=True)


def main(data_dir, label_dir, out_dir):
    """
    Extract frames from videos avaialble in the original DAiSEE dataset.
    Extracted frames will be stored as jpeg images.

    Args:
        data_dir: Path to directory which contains videos.
            This directory will contain Test, Train and Validation
            subdirectory, which has same structure as downloaded from
            the DAiSEE website.
        label_dir: Path to directory which contains three csv files
            namely TrainLabels.csv, TestLabels.csv, ValidationLabels.csv
        out_dir: Path to directory which will be used to store extracted
            frames. If the directory is not already present it will be
            created.
    """
    data_dir = Path(r"DataSet")
    label_dir = Path(r"Labels")
    out_dir = Path(r"Frames")

    subdirectories = ["Train", "Test", "Validation", "FinalTest"]
    for subdirectory in subdirectories:
        if subdirectory == "FinalTest":
            sdir = data_dir / "Test"
            label_path = str(label_dir) + "/TestLabels.csv"
        else:
            sdir = data_dir / subdirectory
            label_path = str(label_dir) + f"/{subdirectory}Labels.csv"
        odir = out_dir / subdirectory
        odir.mkdir(parents=True, exist_ok=True)
        label = pd.read_csv(label_path)
        print(f"Extracting frames for {subdirectory}")
        with ProgressBar(maxval=len(list(sdir.glob("*/*/*")))) as bar:
            for i, video in enumerate(sdir.glob("*/*/*")):
                if label['ClipID'].str.contains(video.parts[-1]).any():
                    get_frames(subdirectory, video, odir)
                bar.update(i)


