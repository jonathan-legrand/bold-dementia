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

def compute_and_cache_ts(atlas:Atlas, bids_dir:Path, ppath:Path):
    psuffix = ppath.stem
    
    memento = Memento(
        bids_dir,
        ppath,
        atlas=atlas,
        cache_dir=bids_dir / "derivatives" / (atlas.name + "_" + psuffix),
    )
    if atlas.is_soft:
        print("is_soft is True, default to serial caching")
        memento.cache_series() # For some reason parallel caching is slow with soft atlases
    else:
        print("Using parallel caching")
        memento.parallel_caching(n_jobs=8)
    
import sys

if __name__ == "__main__":
    is_soft = eval(sys.argv[2])
    if not isinstance(is_soft, bool):
        raise TypeError("is_soft should be in {True, False}")
    atlas = Atlas.from_name(sys.argv[1], soft=eval(sys.argv[2]))
    try:
        pname = sys.argv[3]
    except IndexError:
        print("Using default merged phenotypes")
        pname = "merged_phenotypes.csv"
    
    ppath = Path(config["data_dir"]) / pname
    print(f"Using phenotypes from {ppath}")

    bids_dir = Path(config["bids_dir"])
    
    compute_and_cache_ts(atlas, bids_dir, ppath)