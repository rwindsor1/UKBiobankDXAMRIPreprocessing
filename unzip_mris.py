import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from utils import get_scan_in_zip
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Unzip and stitch Dixon MRIS from UK Biobank')
parser.add_argument('-i','--input_path',help='The path to the downloaded raw MRIs')
parser.add_argument('-o','--output_path',help='The path to save the processed MRIs')
parser.add_argument('--thread_idx',default=0,type=int,help='The fractile to start processing the MRIs')
parser.add_argument('--num_threads',default=1,type=int,help='The fractile to end processing the MRIs')

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path


mris = sorted(glob.glob(os.path.join(input_path,'*.zip')))

start_idx = int(np.floor(len(mris)*args.thread_idx/args.num_threads))
end_idx = int(np.ceil(len(mris)*(args.thread_idx+1)/args.num_threads))
print(start_idx, end_idx)

print(f"Found {len(mris)} MRIs!")

for idx in tqdm(range(start_idx, end_idx+1)):
    try:
        mri = mris[idx]
        output_dir = mri.replace(input_path, output_path)
        scan_output_path = os.path.join(output_path, os.path.basename(mri).replace('.zip','.pkl'))
        if os.path.exists(scan_output_path): continue
        fat_volume = get_scan_in_zip(mri, 'F')
        water_volume = get_scan_in_zip(mri, 'W')
        output_dict = {'fat_scan': fat_volume['scan'].transpose(1,0,2)[:,:,::-1],
                       'water_scan':water_volume['scan'].transpose(1,0,2)[:,:,::-1],
                       'pixel_spacing': [fat_volume['slice_thickness']]*3}

        with open(scan_output_path,'wb') as f:
            pickle.dump(output_dict, f)
    except Exception as E:
        print(E)
        continue

