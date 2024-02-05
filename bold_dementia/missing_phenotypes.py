"""
Show and export subjects which have an fMRI but no phenotype
"""

from pathlib import Path
from bold_dementia.connectivity.atlases import Atlas
import pandas as pd
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from bold_dementia.data.memento import Memento


def missing_phenotypes(bids_dir, phenotypes_path):
    fmri_path = get_bids_files(
        bids_dir / "derivatives/fmriprep-23.2.0",
        "bold",
        modality_folder="func",
        file_type="nii.gz",
        filters=[
            ("space", "MNI152NLin6Asym")
        ],
    )
    df = pd.DataFrame(map(parse_bids_filename, fmri_path))
    
    phenotypes = Memento.load_phenotypes(
        phenotypes_path,
        augmented=True
    )
    phenotypes["sub"] = phenotypes["NUM_ID"].map(lambda x: x[4:])
    
    m = pd.merge(
        left=df,
        right=phenotypes,
        how="left",
        on="sub"
    )
    missing_phenotypes = m[m["CEN_ANOM"].isna()]["sub"]
    return pd.Series(("SUBJ" + missing_phenotypes).unique())

if __name__ == "__main__":
    BIDSDIR = Path(
        "/georges/memento/BIDS"
    )
    PPATH = Path(
        "/bigdata/jlegrand/data/Memento/output/augmented_phenotypes.csv"
    )
    OUTPUT_PATH = "/bigdata/jlegrand/data/Memento/output/missing_phenotypes.csv"

    missing = missing_phenotypes(BIDSDIR, PPATH)
    print(missing)
    missing.to_csv(OUTPUT_PATH)