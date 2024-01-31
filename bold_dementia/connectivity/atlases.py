from sklearn.utils import Bunch
from nilearn import plotting
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker


import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from nilearn import datasets
import pandas as pd

def fetch_atlas_rsn45(
        atlas_tsv="/bigdata/jlegrand/data/Memento/atlas/RSN_M5_clean2_ws.dat",
        atlas_path="/bigdata/jlegrand/data/Memento/atlas/RSN_N41_atlas_M5_clean2_wscol.nii"
    ):
    df = pd.read_csv(atlas_tsv, sep="\t")
    labels = df["tissue"].map(lambda x: x.strip()) + "_" + df["nroi"].astype(str) + "_" + "RSN" + df["RSN"].astype(str)
    atlas_bunch = Bunch(
        maps=atlas_path,
        labels=labels,
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
    "rsn41": fetch_atlas_rsn45,
    "harvard-oxford": lambda : datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"),
    "schaeffer": lambda : datasets.fetch_atlas_schaefer_2018(resolution_mm=2),
    "difumo": lambda : datasets.fetch_atlas_difumo(legacy_format=False),
    "smith": datasets.fetch_atlas_smith_2009,
    "msdl": datasets.fetch_atlas_msdl
}


class Atlas(Bunch):
    @classmethod
    def from_kwargs(cls, name, soft, **atlas_kwargs) -> None:
        new = cls(**atlas_kwargs)

        new.is_soft = soft
        new.name = name
        return new
    
    @classmethod
    def from_name(cls, name, soft):
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
                **mask_kw
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
            