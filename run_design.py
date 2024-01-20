import sys
import os
import argparse

'''currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)'''

from GPD.utils import *
def main(args):
    pdb = args.pdb_file
    num_seq = args.num_seq
    out_folder = args.out_folder
    if args.fixed_positions:
        fixed_positions = args.fixed_positions
        example_dict = {}
        with open(fixed_positions) as f:
            for line in f:
                each = line.strip().split("\t")
                example_dict[int(each[0])-1] = each[1]
        print('fixed positions')
        recovery, length = run_single_protein(output_dir=out_folder, pdb_name=pdb, times=int(num_seq), design_dict=example_dict, create_fasta=True)
    else:
        recovery, length = run_single_protein(output_dir=out_folder, pdb_name=pdb, times=int(num_seq), design_dict=None, create_fasta=True)
    print("Mean Recovery\t%.3f\tThe Length of PDB File\t%d" %(float(recovery), length))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="fixed-backbone design by GPD.")

    argparser.add_argument("--pdb_file", type=str, help="Path to a single PDB to be designed")
    argparser.add_argument("--num_seq", type=int, default=2, help="Number of sequences to generate per target")
    argparser.add_argument("--out_folder", type=str, default='./', help="Path to a folder to output sequences")
    argparser.add_argument("--fixed_positions", type=str, default='', help="Path to a file with fixed positions")
    args = argparser.parse_args()    
    main(args)   


