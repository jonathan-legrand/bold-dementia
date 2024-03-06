"""
Our model uses average time series from a given parcellation as input features.
Image loading and parcellation take too much time to be integrated
in the training input pipeline, so it is performed before and
cached in a dedicated directory.
"""

from pathlib import Path
from data.memento import Memento
from connectivity.atlases import fetch_aicha
from bold_dementia import get_config
from nilearn.datasets import fetch_atlas_harvard_oxford
from connectivity.atlases import Atlas


config = get_config()
BIDSDIR = Path(config["bids_dir"])
PPATH = Path(config["augmented_phenotypes"])


# TODO Pass arguments from argv
def compute_and_cache_ts(atlas):
    memento = Memento(
        BIDSDIR,
        PPATH,
        atlas=atlas,
        cache_dir=BIDSDIR / "derivatives" / atlas.name
    )
    if atlas.is_soft:
        print("is_soft is True, default to serial caching")
        memento.cache_series() # For some reason parallel caching is slow with soft atlases
    else:
        print("Using parallel caching")
        memento.parallel_caching(n_jobs=8)
    

# TODO Pass atlas (and destination and ?) as argument to command line
import sys

if __name__ == "__main__":
    is_soft = eval(sys.argv[2])
    if not isinstance(is_soft, bool):
        raise TypeError("is_soft should be in {True, False}")
    atlas = Atlas.from_name(sys.argv[1], soft=eval(sys.argv[2]))
    compute_and_cache_ts(atlas)