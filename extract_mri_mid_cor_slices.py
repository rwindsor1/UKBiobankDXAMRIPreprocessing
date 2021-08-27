import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from utils import get_scan_in_zip
import matplotlib.pyplot as plt
from med_img_utils import *

parser = argparse.ArgumentParser(description='Extract the mid coronal slices from stitched UK Biobank Dixon MR scans')
parser.add_argument('-i','--input_path',help='The path to the downloaded raw DXAs')
parser.add_argument('-o','--output_path',help='The path to save the processed DXAs')

def find_mid_cor_slice(mri):
    cor_slice_intensities = mri['fat_scan'].sum(axis=0).sum(axis=-1)
    mid_corr_slice = np.abs(np.cumsum(cor_slice_intensities)/cor_slice_intensities.sum() - 0.5).argmin()
    import pdb; pdb.set_trace()
    return mid_corr_slice




args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

mris = sorted(glob.glob(os.path.join(input_path,'*.pkl')))
print(f"Found {len(mris)} MRIs!")

for idx, mri_path in enumerate(tqdm(mris)):
    scan = pickle.load(open(mri_path,'rb'))
    output_dir = mri_path.replace(input_path, output_path)
    mid_cor_slice = find_mid_cor_slice(scan)
    # plt.figure(figsize=(10,3))
    # plt.subplot(131)
    # plt.imshow(scan['fat_scan'][:,mid_cor_slice-10].T)
    # plt.subplot(132)
    # plt.imshow(scan['fat_scan'][:,mid_cor_slice].T)
    # plt.subplot(133)
    # plt.imshow(scan['fat_scan'][:,mid_cor_slice+10].T)
    # plt.show()
    out_fat_scan = scan['fat_scan'][:,mid_cor_slice-10:mid_cor_slice+10:2]
    out_water_scan = scan['fat_scan'][:,mid_cor_slice-10:mid_cor_slice+10:2]
    scan['fat_scan'] = out_fat_scan
    scan['water_scan'] = out_fat_scan
    scan['pixel_spacing'][1] = scan['pixel_spacing'][1]*2
    with open(output_dir,'wb') as f:
        pickle.dump(scan,f)

