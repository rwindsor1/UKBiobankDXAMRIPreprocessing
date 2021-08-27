'''
Python script to download DXA Dicoms from the UK Biobank. Requires an
authentication keyfile, list of patient ids and output path as arguments.
'''
import os, glob
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('-a',
                    help="""Path to Biobank authentication file. Defaults to 
                    '.ukbkey''""",
                    default='.ukbkey',
                    )

parser.add_argument('-o',
                    help="""Output path to save downloaded zipped DXA 
                    DICOMS to. """
                    )

parser.add_argument('-e',
                    help="""Path to list of patient ids to download. Defaults 
                    to 'ukbb_eids.txt'""",
                    default='ukbb_eids.txt'
                    )

args = parser.parse_args()

curr_path = os.path.dirname(os.path.abspath(__file__))
output_path = args.o
eids_path = args.e
auth_path = args.a

# open eids file
with open(eids_path, 'r') as f:
    eids = f.read().splitlines()

# make bulk file for DXA DICOMs from eids
bulk_list = list(map(lambda x: x + ' 20201_2_0', eids))
num_files = len(bulk_list)
bulk_path = os.path.join(curr_path,Path(eids_path).stem + '.mribulk')
with open(bulk_path,'w') as f:
    f.write('\n'.join(bulk_list))

# change directory to output path and run ukbfetch
# note this can only be done with 1000 scans at a time.

os.chdir(output_path)
for i in range((num_files//1000)+1):
    os.system(f"{os.path.join(curr_path,'ukbfetch')} -a{auth_path} -b{bulk_path} -s{i*1000} -m1000")
