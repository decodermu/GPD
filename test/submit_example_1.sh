source activate GPD
input_pdb="./inputs/1tca.pdb"
output_dir="./outputs/example_1_outputs"
python ../run_design.py \
        --pdb_file $input_pdb \
        --num_seq  2 \
        --out_folder $output_dir
