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


def compute_and_cache_ts(atlas):
    memento = Memento(
        BIDSDIR,
        PPATH,
        atlas=atlas,
        cache_dir=BIDSDIR / "derivatives" / atlas.name
    )
    memento.parallel_caching(n_jobs=4)
    

# TODO Pass atlas (and destination?) as argument to command line
if __name__ == "__main__":
    atlas = Atlas.from_name("msdl", soft=True)
    compute_and_cache_ts(atlas)