# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data.memento import Memento, MementoTS
from connectivity.atlases import fetch_aicha
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn import plotting
from torch.utils.data import DataLoader


BIDSDIR = Path("/georges/memento/BIDS")

# %%
tst = MementoTS()
# %%
aicha = fetch_aicha()
ho = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")


# %%
m = Memento(BIDSDIR, "data/phenotypes.tsv")
dl = DataLoader(m, batch_size=1, shuffle=False)
# %%
op = next(iter(dl))
# %%

#plt.plot(ts)
#plt.title(f"Demented is {demented}")
#plt.show()
# %%

import cProfile
cProfile.run('m.__getitem__(34)')
# %%
# %%
phenotypes = Memento.load_phenotypes("data/phenotypes.tsv")
# %%
from nilearn.interfaces.bids import parse_bids_filename
pd.DataFrame(map(parse_bids_filename, Path("dataset_cache/time_series").iterdir()))
# %%
