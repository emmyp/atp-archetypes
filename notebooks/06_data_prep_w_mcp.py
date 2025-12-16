# %% load and filter
from pathlib import Path

import numpy as np
import pandas as pd

raw_dir = Path("../data/raw")

matches = pd.read_csv(raw_dir / "charting-m-matches.csv")
oview = pd.read_csv(raw_dir / "charting-m-stats-Overview.csv")
serve = pd.read_csv(raw_dir / "charting-m-stats-ServeBasics.csv")
servedir = pd.read_csv(raw_dir / "charting-m-stats-ServeDirection.csv")
ret = pd.read_csv(raw_dir / "charting-m-stats-ReturnOutcomes.csv")
retdep = pd.read_csv(raw_dir / "charting-m-stats-ReturnDepth.csv")
shotdir = pd.read_csv(raw_dir / "charting-m-stats-ShotDirection.csv")
shotty = pd.read_csv(raw_dir / "charting-m-stats-ShotTypes.csv")
rally = pd.read_csv(raw_dir / "charting-m-stats-Rally.csv")
netpts = pd.read_csv(raw_dir / "charting-m-stats-NetPoints.csv")
snv = pd.read_csv(raw_dir / "charting-m-stats-SnV.csv")

matches.head()
