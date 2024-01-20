source activate GPD
input_pdb="../inputs/1tca.pdb"
output_dir="../outputs/example_2_fixed"
fiexed_sites="../inputs/fixed_sites.txt"
python ../protein_GPD_run.py \
        --pdb_file $input_pdb \
        --num_seq  2 \
        --out_folder $output_dir \
        --fixed_positions $fiexed_sites