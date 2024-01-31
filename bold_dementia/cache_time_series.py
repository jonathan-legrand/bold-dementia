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
        cache_dir="/georges/memento/BIDS/derivatives/model_inputs"
    )
    memento.cache_series()
    

# TODO Pass atlas and destination as argument to command line?
if __name__ == "__main__":
    #atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas = Atlas.from_name("harvard-oxford", soft=False)
    compute_and_cache_ts(atlas)