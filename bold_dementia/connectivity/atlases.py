from sklearn.utils import Bunch
from nilearn import plotting
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from pathlib import Path


import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from nilearn import datasets
import pandas as pd
import nibabel as nib
import numpy as np
import os

from bold_dementia import get_config

# TODO This is just a quick fix
config = get_config(Path(os.getcwd()) / "config.yml") # We will need volumes path from conf
config["custom_atlases"] = f"{config['data_dir']}/atlas"


def fetch_atlas_m5n33(
    atlas_csv=f"{config['custom_atlases']}/RSN_N33/flat_m5n33.csv",
    atlas_path=f"{config['custom_atlases']}/RSN_N33/flat_m5n33.nii.gz"
):
    if (Path(atlas_path).exists() and Path(atlas_csv).exists()) is False:
        print("Creating flat atlas from RSN33")
        _create_m5n33()
        print("Done")

    df = pd.read_csv(atlas_csv)
    labels = df.RSN.to_list()
    atlas_bunch = Bunch(
        maps=atlas_path,
        labels=labels,
        networks=df.Labels.to_list(), # Shitty choice of names I know
        description="Experimental atlas of resting state networks"
    )
    return atlas_bunch
        

def _create_m5n33(atlas_path=Path(f"{config['custom_atlases']}/RSN_N33/RSN_Cog33_NoOverlap.nii")):
    """This has only been called once to create the 3D version of RSN_N33

    Args:
        atlas_path (_type_, optional): _description_. Defaults to Path(f"{config['custom_atlases']}/RSN_N33/RSN_Cog33_NoOverlap.nii").

    Returns:
        _type_: _description_
    """
    
    img = nib.load(atlas_path)
    
    # We are going to be VERY specific about using
    # int32 because lots of neuroimaging softwares dislike int64
    # which is numpy's default.
    flat_m5 = np.zeros(img.shape[:3], dtype=np.int32)
    
    for network_idx in range(img.shape[-1]):
        network_mask = img.slicer[..., network_idx].get_fdata()
        network_mask = np.where(network_mask !=0, network_idx + 1, 0) # Value 0 is for background
        flat_m5 += network_mask.astype(np.int32)


    mapping = pd.read_csv("/homes_unix/jlegrand/data/Memento/atlas/RSN_N33/RSN41_cognitive_labeling.csv").rename(columns={"Numbering_new": "RSN"})
    mapping.loc[mapping["Labels"].isna(), "Labels"] = "Unknown"
    mapping["color_code"] = range(1, 34)

    output_dir = atlas_path.parent
    network_maps = nib.Nifti1Image(flat_m5, img.affine)
    nib.save(network_maps, output_dir / "flat_m5n33.nii.gz")
    mapping.to_csv(output_dir / "flat_m5n33.csv")
    
    return output_dir, mapping

    
trunc_indices = (445, 444, 447, 446, 442, 441, 443, 440, 448)

def create_m5_notrunc_(old_atlas_path=Path("/bigdata/jlegrand/data/Memento/atlas/RSN_N41_atlas_M5_clean2_wscol.nii")):
    img = nib.load(old_atlas_path)
    arr = img.get_fdata()
    
    new_arr = np.where(np.isin(arr, trunc_indices), 0, arr)

    u = np.unique(new_arr)
    derivative = u[1:] - u[:-1]
    assert np.all(derivative == 1), "Reindexing failed because of jumps in new array values"
    
    output_path = old_atlas_path.parent / "M5_no-trunc.nii.gz"
    network_maps = nib.Nifti1Image(new_arr.astype(np.int32), img.affine)
    nib.save(network_maps, output_path)
    print(f"New atlas stored in {output_path}")

    
def fetch_atlas_m5n33_regions(
        atlas_tsv=f"{config['custom_atlases']}/RSN_M5_clean2_ws.dat",
        updated_rsn=f"{config['custom_atlases']}/RSN_N33/RSN41_cognitive_labeling_updated.csv",
        atlas_path=f"{config['custom_atlases']}/M5_no-trunc.nii.gz"
    ):
    original_labels = pd.read_csv(atlas_tsv, sep="\t")
    networks = "RSN" + original_labels["RSN"].astype(str).apply(lambda x: x.zfill(2))
    original_labels["Numbering_original"] = networks

    # TODO and simplify please
    notrunc = original_labels.drop(original_labels[original_labels.tissue.str.contains("trunc")].index, axis=0)

    updated_rsn = pd.read_csv(
        f"{config['custom_atlases']}/RSN_N33/RSN41_cognitive_labeling_updated.csv"
    )
    merged = pd.merge(
        notrunc,
        updated_rsn,
        on="Numbering_original",
        how="inner"
    )
    labels = merged["Anatomical label achille 2024"] + "_" + merged["icol"].astype(str).map(lambda x: x.zfill(3))
    labels = labels.to_list()
    
    atlas_bunch = Bunch(
        maps=atlas_path,
        labels=labels,
        networks=merged.Numbering_new.to_list(),
        description="Experimental atlas of resting state networks with regions, v.0.3 with 33 networks",
        **dict(merged)
    )
    return atlas_bunch


def fetch_atlas_rsn41(
        atlas_tsv="/bigdata/jlegrand/data/Memento/atlas/RSN_M5_clean2_ws.dat",
        atlas_path="/bigdata/jlegrand/data/Memento/atlas/RSN_N41_atlas_M5_clean2_wscol.nii"
    ):
    df = pd.read_csv(atlas_tsv, sep="\t")
    networks = "RSN" + df["RSN"].astype(str).apply(lambda x: x.zfill(2))
    labels = df["tissue"].map(lambda x: x.strip()) + "_" + df["nroi"].astype(str) + "_" + networks
    print(labels)
    atlas_bunch = Bunch(
        maps=atlas_path,
        labels=labels.to_list(),
        networks=networks.to_list(),
        description="Experimental atlas of resting state networks"
    )
    return atlas_bunch


def fetch_aicha(
    atlas_xml="/homes_unix/jlegrand/AD-prediction/AICHA/AICHA.xml",
    atlas_path="/homes_unix/jlegrand/AD-prediction/AICHA/AICHA.nii"
):
    
    atlas_desc = ET.parse(atlas_xml)
    label_search = atlas_desc.findall(".//label")
    labels = [l[1].text for l in label_search]
    
    atlas_bunch = Bunch(
        maps=atlas_path,
        labels=labels,
        indices=list(range(len(labels))),
        description="AICHA parcellation"
    )
    
    return atlas_bunch


def make_overlay_slices(volume):
    h, w, d = volume.shape
    return (
            volume[h//2 - 20, :, :],
            volume[:, w//2, :],
            volume[:, :, d//2]
            )
    
def overlay_atlas(img, atlas):
    
    """ Function to display column of image slices
    with atlas overlay
    """
    img_slices = make_overlay_slices(img)
    atlas_slices = make_overlay_slices(atlas)
    names = ("saggital", "coronal", "axial")
    fig, axes = plt.subplots(len(img_slices), figsize=(10, 10))
    
    for i, (slice_img, slice_atlas) in enumerate(zip(img_slices, atlas_slices)):
        axes[i].imshow(slice_img.T, cmap="gray", origin="lower")
        axes[i].imshow(slice_atlas.T, cmap="seismic", origin="lower", alpha=0.5)
        axes[i].set_title(names[i])
        
    return fig

atlas_mapping = {
    "AICHA": fetch_aicha,
    "rsn41": fetch_atlas_rsn41, # Keep former name for compatibility reasons
    "gillig": fetch_atlas_m5n33,
    "GINNA": fetch_atlas_m5n33, # Always gillig in my heart
    "gillig-regions": fetch_atlas_m5n33_regions,
    "harvard-oxford": lambda : datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"),
    "schaeffer": lambda : datasets.fetch_atlas_schaefer_2018(resolution_mm=2),
    "schaeffer200": lambda : datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2),
    "schaeffer100": lambda : datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2),
    "difumo": lambda : datasets.fetch_atlas_difumo(legacy_format=False),
    "smith": datasets.fetch_atlas_smith_2009,
    "msdl": datasets.fetch_atlas_msdl
}

is_soft_mapping = {
    "schaeffer200": False,
    "msdl": True,
    "GINNA": False
}

class Atlas(Bunch):
    @classmethod
    def from_kwargs(cls, name, soft, **atlas_kwargs) -> None:
        new = cls(**atlas_kwargs)

        new.is_soft = soft
        new.name = name
        return new
    
    @classmethod
    def from_name(cls, name, soft=None):
        if soft is None:
            soft = is_soft_mapping[name] 
        atlas_kwargs = atlas_mapping[name]()
        new = cls(**atlas_kwargs)
        new.is_soft = soft
        new.name = name
        new.labels_ = atlas_kwargs["labels"]
        return new

    def get_coords(self):
        if "region_coords" in self.keys():
            return self.region_coords
        elif self.is_soft:
            return plotting.find_probabilistic_atlas_cut_coords(self.maps)
        else:
            return plotting.find_parcellation_cut_coords(self.maps)
    
    def overlay(self):
        raise NotImplementedError()

    def plot(self, **plotting_kwargs):
        if self.is_soft:
            return plotting.plot_prob_atlas(self.maps, title=self.name, **plotting_kwargs)
        else:
            return plotting.plot_roi(self.maps, title=self.name, **plotting_kwargs)

    def fit_masker(self, **masker_kw):
        if self.is_soft:
            masker = NiftiMapsMasker(
                maps_img=self.maps,
                **masker_kw
            )
        else:
            masker = NiftiLabelsMasker(
                labels_img=self.maps,
                #labels=self.labels, # TODO Test that
                **masker_kw
            )
        masker.fit()
        return masker

    @property
    def labels(self):
        # The issue is that for difumo the labels
        # are in a data frame
        # Check type of labels instead?
        if self.name == "difumo":
            return self.labels_.difumo_names.to_list()
        else:
            return self.labels_

    @property
    def macro_labels(self):
        if "networks" in self.keys():
            return self.networks
        l = self.labels
        return list(map(lambda x: str(x).split("_")[2], l))
            