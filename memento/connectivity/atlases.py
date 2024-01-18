from sklearn.utils import Bunch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from nilearn import datasets



def fetch_aicha(
    atlas_xml="AICHA/AICHA.xml",
    atlas_path="AICHA/AICHA.nii"
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
    "harvard-oxford": lambda : datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"),
    "schaeffer": lambda : datasets.fetch_atlas_schaefer_2018(resolution_mm=2),
    "difumo": lambda : datasets.fetch_atlas_difumo(legacy_format=False),
    "smith": datasets.fetch_atlas_smith_2009,
    "msdl": datasets.fetch_atlas_msdl
}