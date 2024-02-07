"""
Our model uses average time series from a given parcellation as input features.
Image loading and parcellation take too much time to be integrated
in the training input pipeline, so it is performed before and
cached in a dedicated directory.
"""


from pathlib import Path
from data.memento import Memento
from connectivity.atlases import fetch_aicha
from nilearn.datasets import fetch_atlas_harvard_oxford
from connectivity.atlases import Atlas

BIDSDIR = Path("/georges/memento/BIDS")
PPATH = Path("/bigdata/jlegrand/data/Memento/output/augmented_phenotypes.csv")


def compute_and_cache_ts(atlas):
    memento = Memento(
        BIDSDIR,
        PPATH,
        atlas=atlas,
        cache_dir="/georges/memento/BIDS/derivatives/rsn41"
    )
    memento.parallel_caching()
    

# TODO Pass atlas and destination as argument to command line?
if __name__ == "__main__":
    atlas = Atlas.from_name("rsn41", soft=False)
    compute_and_cache_ts(atlas)