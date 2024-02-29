"""
Atlas visualisation : stack ROI on 4th axis
"""
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import itertools
import sys
from bold_dementia.utils.iterables import unique
from bold_dementia import get_config
from bold_dementia.connectivity import Atlas

config = get_config()

def parse_command_line():
    try:
        atlas_name = sys.argv[1]
        is_soft = eval(sys.argv[2])
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <atlas> <is_soft>")
    except NameError:
        raise SystemExit(
            f"Usage: is_soft must be True or False, you passed {sys.argv[2]}")

    return atlas_name, is_soft


if __name__ =="__main__":
    atlas_name, is_soft = parse_command_line()

    atlas = Atlas.from_name(atlas_name, soft=is_soft)
    output_path = Path(config["output_dir"]) / "atlases" / f"{atlas.name}_overlay.nii.gz"
    
    network_images = []
    
    try:
        labels = atlas.macro_labels
    except IndexError:
        print("No macro labels found, overlaying base labels")
        labels= atlas.labels

    networks = np.array(labels)
    maps = nib.load(atlas.maps)
    img = maps.get_fdata()
    
    for network in unique(labels):
        
        network_colors = list(np.where(networks == network)[0])
        
        print(f"{network} : {len(network_colors)} regions")
    
        network_img = np.where(np.isin(img, network_colors), 1., 0.)
        bg_mask = np.where(img == 0, 0, 1)
        network_images.append(network_img * bg_mask)
    
    output_img = np.stack(network_images, axis=-1)
    network_maps = nib.Nifti1Image(output_img, maps.affine)
    nib.save(network_maps, output_path)
    