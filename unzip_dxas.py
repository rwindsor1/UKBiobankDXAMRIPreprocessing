import sys,os,glob
import argparse
import zipfile
import shutil
import pydicom
from tqdm import tqdm
from med_img_utils import *
import matplotlib.pyplot as plt
import pickle as pkl


parser = argparse.ArgumentParser(description='Unzip and Process DXAs from UKBB')
parser.add_argument('-i','--input_path',help='The path to the downloaded raw DXAs')
parser.add_argument('-o','--output_path',help='The path to save the processed DXAs')

args=parser.parse_args()

input_path = args.input_path
output_path = args.output_path
dxas = sorted(glob.glob(os.path.join(input_path,'*.zip')))

print(f"Found {len(dxas)} DXAs!")

for idx, dxa in enumerate(tqdm(dxas)):
    output_dir =dxa.replace(input_path, output_path).replace('.zip','')
    shutil.unpack_archive(dxa,output_dir)

    dcm_paths = glob.glob(os.path.join(output_dir,'*.dcm'))
    dcm_objs = [pydicom.dcmread(dcm_path) for dcm_path in dcm_paths]
    dcm_objs = sorted([dcm_obj for dcm_obj in dcm_objs if dcm_obj.ProtocolName == 'Total Body'],key=lambda x: int(x.InstanceNumber))
    tissue_dxa = dcm_objs[0]
    pixel_spacing = [tissue_dxa.ExposedArea[1]/tissue_dxa.Rows,
                     tissue_dxa.ExposedArea[0]/tissue_dxa.Columns]
    bone_dxa = dcm_objs[1]
    out_dict = {'tissue': tissue_dxa.pixel_array,'bone': bone_dxa.pixel_array,
                'pixel_spacing':pixel_spacing}
    with open(os.path.join(output_dir, os.path.basename(dxa).replace('.zip','.pkl')),'wb') as f:
        pkl.dump(out_dict, f)
    for x in dcm_paths: os.remove(x)
    if os.path.isfile(os.path.join(output_dir,'manifest.cvs')): 
        os.remove(os.path.join(output_dir,'manifest.cvs'))


    





