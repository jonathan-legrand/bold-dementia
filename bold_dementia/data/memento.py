import torch
from torch.utils.data import Dataset, DataLoader

from typing import List
from pathlib import Path
import pandas as pd
import nibabel as nib
import joblib
import os

from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds


session_mapping = {
    "IRM_M0": "M000",
    "IRM_M24": "M024",
    "IRM_M48": "M048"
}

# TODO handle background in atlases
# TODO Use caching for bids indexing
# TODO Type annotations
class Memento(torch.utils.data.Dataset):
    def __init__(
        self,
        bids_path,
        phenotypes_path,
        atlas=None,
        cache_dir="dataset_cache"
    ):
        
        # TODO Check atlas consistency when loading from cache
        if atlas is None:
            atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        for key, value in atlas.items():
            setattr(self, key, value)

        self.scans_ = self.index_bids(bids_path)
        self.phenotypes_ = self.load_phenotypes(phenotypes_path)


        self.masker = NiftiLabelsMasker(
            labels_img=self.maps,
            standardize="zscore_sample"
        )

        fmri_path = self.scans_.file_path.to_list()
        self.confounds, self.sample_mask = load_confounds(
            fmri_path,
            strategy=["high_pass", "motion", "wm_csf"],
            motion="basic",
            wm_csf="basic",
        )

        self.masker.fit(
            fmri_path
        )

        self.cache_dir = Path(cache_dir)

    
    # TODO Change default
    @staticmethod
    def load_phenotypes(ppath, augmented=True):
        if augmented:
            phenotypes = pd.read_csv(
                ppath,
                index_col=0
            )
            format = None
        else:
            phenotypes = pd.read_csv(
                ppath,
                sep="\t",
                encoding="unicode_escape",
            )
            format = "%d/%m/%Y"

        # I don't trust date parsing in read_csv
        for date_col in {
                "DEMENCE_DAT",
                "INCCONSDAT_D",
                "IRM_M0",
                "IRM_M24",
                "IRM_M48"
            }:
            phenotypes[date_col] = pd.to_datetime(
                phenotypes[date_col],
                format=format
            )

        return phenotypes.rename(columns=session_mapping)

    @staticmethod
    def index_bids(bids_dir):
        fmri_paths = get_bids_files(
        bids_dir / "derivatives/fmriprep-23.2.0",
        "bold",
        file_type="nii.gz",
        filters=[
            ("space", "MNI152NLin6Asym")
            ],
        )
        scans = pd.DataFrame(map(parse_bids_filename, fmri_paths))

        return scans

    def _extract_ts(self, idx):
        fmri_path = self.scans_.loc[idx, "file_path"]
        
        # For some reason passing the path
        # is slower than loading the image
        # using nibabel, although the masker
        # accepts both
        fmri = nib.load(fmri_path)

        time_series = self.masker.transform(
            fmri,
            confounds=self.confounds[idx],
            sample_mask=self.sample_mask[idx]
        )
        return time_series
    
    def __getitem__(self, idx):
        if idx > len(self):
            raise KeyError()
        time_series = self._extract_ts(idx)
        return time_series, self.is_demented(idx)

    def __getitems__(self, indices:List[int]):
        time_series = self._extract_ts(indices)
        return [(time_series, self.is_demented(idx)) for idx in indices]

    def __len__(self):
        return len(self.scans_)

    def is_demented(self, idx):
        session = self.scans_.loc[idx, "ses"]
        scan_date = self.phenotypes_.loc[idx, session]
        demence_date = self.phenotypes_.loc[idx, "DEMENCE_DAT"]
        return scan_date > demence_date

    def cache_series(self):
        if not os.path.exists(self.cache_dir / "time_series"):
            os.makedirs(self.cache_dir / "time_series")
            
        for idx, row in self.scans_.iterrows():
            fpath = f"{self.cache_dir}/time_series/{row.file_basename}"
            print(row.file_basename)
            
            # We do not overwrite, if you want a new cache
            # delete the former one yourself
            if not os.path.exists(fpath):
                ts = self._extract_ts(idx)
                joblib.dump(ts, fpath)

        self.phenotypes_.to_csv(f"{self.cache_dir}/phenotypes.csv")
        
class MementoTS(Memento):
    def __init__(self, cache_dir="dataset_cache"):
        cache = Path(cache_dir)
        self.phenotypes_ = pd.read_csv(
            cache / "phenotypes.csv",
            index_col=0,
            parse_dates=True,
            date_format="%Y/%m/%d"
        )
        self.time_series = []
        scans = []
        for fpath in (cache / "time_series").iterdir():
            ts = joblib.load(fpath)
            self.time_series.append(ts)
            scans.append(parse_bids_filename(fpath))
            self.scans_ = pd.DataFrame(scans)

    def __getitem__(self, idx):
        return self.time_series[idx], self.is_demented(idx)