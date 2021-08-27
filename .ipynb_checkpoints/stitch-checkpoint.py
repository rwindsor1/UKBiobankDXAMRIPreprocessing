import os
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from utils import get_scan_in_zip
import matplotlib.pyplot as plt

scans = pd.read_csv('biobank_mri_dxa_id.csv')
seq = 'F' # F, W, in, opp
mri_root_path = '/scratch/shared/beeond1/amirj/UKBiobank/20201/'
save_path = '/scratch/shared/beeond1/amirj/UKBiobank/stitched_mris'

scan_num = scans.shape[0]

for index, row in tqdm(scans.iterrows(), total=scan_num):
    try:
        curr_mri_file_name = row['mri_filename']
        volume = get_scan_in_zip(mri_root_path + curr_mri_file_name + '.zip', seq)
        scan = np.moveaxis(volume['scan'], -1, 0)
        scan = np.flip(scan,axis=0)
        scan = (scan - np.min(scan))/np.ptp(scan)

        res = {}
        res['volume'] = scan
        res['pixel_spacing'] = [volume['slice_thickness'], volume['slice_thickness']]
        res['mri_filename'] = curr_mri_file_name
        res['dxa_filename'] = row['dxa_filename']
        res['id'] = row['id']
        res['last_name'] = row['last_name']
        res['seq'] = seq

        with open(os.path.join(save_path, f"{curr_mri_file_name}_{seq}.pkl"), 'wb') as f:
            pickle.dump(res, f)
    except Exception as e:
        curr_mri_file_name = row['mri_filename']
        with open('failed_stitches.log', 'a+') as f:
            f.write(f'{curr_mri_file_name}_{seq}, {e}\n')
                
        
