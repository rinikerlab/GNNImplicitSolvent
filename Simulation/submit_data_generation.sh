
# job1=$(sbatch --parsable --array=[0-1000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job1 --array=[0-1000]%250 -n 4 --time=2:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# job2=$(sbatch --dependency=afterany:$job1 --parsable --array=[1000-2000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job2 --array=[1000-2000]%250 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# job3=$(sbatch --dependency=afterany:$job2 --parsable --array=[2000-3000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job3 --array=[2000-3000]%250 -n 4 --time=2:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# job4=$(sbatch --dependency=afterany:$job3 --parsable --array=[3000-4000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job4 --array=[3000-4000]%250 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# job5=$(sbatch --dependency=afterany:$job4 --parsable --array=[4000-5000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job5 --array=[4000-5000]%250 -n 4 --time=2:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# job6=$(sbatch --dependency=afterany:$job5 --parsable --array=[5000-6000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job6 --array=[5000-6000]%250 -n 4 --time=2:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# job7=$(sbatch --dependency=afterany:$job6 --parsable --array=[6000-7000]%250 -n 8 --time=4:00:00 --tmp=5000 --mem-per-cpu=2000 -o Caculated_caches/slurm_log/cache_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy --cacheonly -nc 8')
# sbatch --dependency=afterany:$job7 --array=[6000-7000]%250 -n 4 --time=2:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'
# sbatch --array=7000-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)Cl" -r 161311 -f generation_smiles/canonical_smiles.npy'

# # Submit TIP5P jobs
# sbatch --array=0-5000:10 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_TIP5P_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "TIP5P" -r 161311 -f generation_smiles/canonical_smiles.npy'
# sbatch --array=5000-7000:10 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_TIP5P_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "TIP5P" -r 161311 -f generation_smiles/canonical_smiles.npy'
# sbatch --array=7000-7411:10 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_TIP5P_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "TIP5P" -r 161311 -f generation_smiles/canonical_smiles.npy'

# # Submit Methanol jobs
# sbatch --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_CO_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CO" -r 161311 -f generation_smiles/canonical_smiles.npy'
# sbatch --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_CO_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CO" -r 161311 -f generation_smiles/canonical_smiles.npy'

# # Submit DMSO jobs
# sbatch --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_DMSO_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CS(=O)C" -r 161311 -f generation_smiles/canonical_smiles.npy'
# sbatch --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/slurm_log/run_generation_DMSO_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CS(=O)C" -r 161311 -f generation_smiles/canonical_smiles.npy'

# Submit DMPU jobs
# job1=$(sbatch --parsable --array=0-7411 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_DMPU_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=C1N(C)CCCN1C" -r 1613111 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/DMPU_starting_coordinates/')
# sbatch --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/DMPU/slurm_log/run_generation_DMPU_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=C1N(C)CCCN1C" -r 1613111 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DMPU_starting_coordinates/ -sl Calculated_data/DMPU/'
# sbatch --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/DMPU/slurm_log/run_generation_DMPU_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=C1N(C)CCCN1C" -r 1613111 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DMPU_starting_coordinates/ -sl Calculated_data/DMPU/'

## List of solvent to submit
# Wasser			O			18	1.0
# Chloroform		ClC(Cl)Cl		119.38	1.49	
# Methanol		CO			32.04	0.792
# DMSO			CS(=O)C			78.13	1.1
# DMPU			O=C1N(C)CCCN1C		128.175	1.06

# # Diethylether		CCOCC			74.123	0.7134
# job2=$(sbatch --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Diethylether_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCOCC" -r 1613112 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Diethylether_starting_coordinates/')

# # use --oneonly to only generate one trajectory per starting structure
# sbatch --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Diethylether/slurm_log/run_generation_Diethylether_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCOCC" -r 1613112 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Diethylether_starting_coordinates/ -sl Calculated_data/Diethylether/ --oneonly'
# sbatch --array=5000-7411 -n 4 --time=24:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Diethylether/slurm_log/run_generation_Diethylether_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCOCC" -r 1613112 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Diethylether_starting_coordinates/ -sl Calculated_data/Diethylether/ --oneonly'

# # Ethanol			OCC			46.069	0.78945
# mkdir /cluster/project/igc/kpaul/Ethanol_starting_coordinates
# job3=$(sbatch --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Ethanol_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "OCC" -r 1613113 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Ethanol_starting_coordinates/')
# sbatch --dependency=afterany:$job3 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Ethanol/slurm_log/run_generation_Ethanol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "OCC" -r 1613113 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Ethanol_starting_coordinates/ -sl Calculated_data/Ethanol/ --oneonly'
# sbatch --dependency=afterany:$job3 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Ethanol/slurm_log/run_generation_Ethanol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "OCC" -r 1613113 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Ethanol_starting_coordinates/ -sl Calculated_data/Ethanol/ --oneonly'

# # DMF			CN(C)C=O		73.095	0.948
# mkdir /cluster/project/igc/kpaul/DMF_starting_coordinates
# job4=$(sbatch --dependency=afterany:$job3 --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_DMF_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CN(C)C=O" -r 1613114 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/DMF_starting_coordinates/')
# sbatch --dependency=afterany:$job4 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/DMF/slurm_log/run_generation_DMF_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CN(C)C=O" -r 1613114 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DMF_starting_coordinates/ -sl Calculated_data/DMF/ --oneonly'
# sbatch --dependency=afterany:$job4 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/DMF/slurm_log/run_generation_DMF_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CN(C)C=O" -r 1613114 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DMF_starting_coordinates/ -sl Calculated_data/DMF/ --oneonly'

# # DCM			ClCCl			84.93	1.3266
# mkdir /cluster/project/igc/kpaul/DCM_starting_coordinates
# job5=$(sbatch --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_DCM_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClCCl" -r 1613115 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/DCM_starting_coordinates/')
# sbatch --dependency=afterany:$job5 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/DCM/slurm_log/run_generation_DCM_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClCCl" -r 1613115 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DCM_starting_coordinates/ -sl Calculated_data/DCM/ --oneonly'
# sbatch --dependency=afterany:$job5 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/DCM/slurm_log/run_generation_DCM_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClCCl" -r 1613115 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DCM_starting_coordinates/ -sl Calculated_data/DCM/ --oneonly'

# # Toluol			Cc1ccccc1		92.141	0.8623
# mkdir /cluster/project/igc/kpaul/Toluol_starting_coordinates
# job6=$(sbatch --dependency=afterany:$job5 --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Toluol_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "Cc1ccccc1" -r 1613116 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Toluol_starting_coordinates/')
# sbatch --dependency=afterany:$job6 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Toluol/slurm_log/run_generation_Toluol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "Cc1ccccc1" -r 1613116 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Toluol_starting_coordinates/ -sl Calculated_data/Toluol/ --oneonly'
# sbatch --dependency=afterany:$job6 --array=5000-7411 -n 4 --time=24:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Toluol/slurm_log/run_generation_Toluol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "Cc1ccccc1" -r 1613116 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Toluol_starting_coordinates/ -sl Calculated_data/Toluol/ --oneonly'

# # Benzol			c1ccccc1		78.114	0.8765
# mkdir /cluster/project/igc/kpaul/Benzol_starting_coordinates
# job7=$(sbatch --dependency=afterany:$job6 --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Benzol_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccccc1" -r 1613117 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Benzol_starting_coordinates/')
# sbatch --dependency=afterany:$job7 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Benzol/slurm_log/run_generation_Benzol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccccc1" -r 1613117 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Benzol_starting_coordinates/ -sl Calculated_data/Benzol/ --oneonly'
# sbatch --dependency=afterany:$job7 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Benzol/slurm_log/run_generation_Benzol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccccc1" -r 1613117 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Benzol_starting_coordinates/ -sl Calculated_data/Benzol/ --oneonly'


# # # Hexan			CCCCCC			86.178	0.6606
# # mkdir /cluster/project/igc/kpaul/Hexan_starting_coordinates
# # job8=$(sbatch --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Hexan_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCCCCC" -r 1613118 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Hexan_starting_coordinates/')
# sbatch --dependency=afterany:$job8 --array=0-5000 -n 4 --time=24:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Hexan/slurm_log/run_generation_Hexan_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCCCCC" -r 1613118 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Hexan_starting_coordinates/ -sl Calculated_data/Hexan/ --oneonly'
# sbatch --dependency=afterany:$job8 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Hexan/slurm_log/run_generation_Hexan_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCCCCC" -r 1613118 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Hexan_starting_coordinates/ -sl Calculated_data/Hexan/ --oneonly'

### CHECK Random SEEDS ###

# # # acetonitrile		CC#N			41.053	0.786
# # mkdir /cluster/project/igc/kpaul/acetonitrile_starting_coordinates
# # job9=$(sbatch --dependency=afterany:$job8 --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_acetonitrile_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC#N" -r 1613119 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/acetonitrile_starting_coordinates/')
# sbatch --dependency=afterany:$job9 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/acetonitrile/slurm_log/run_generation_acetonitrile_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC#N" -r 1613119 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/acetonitrile_starting_coordinates/ -sl Calculated_data/acetonitrile/ --oneonly'
# sbatch --dependency=afterany:$job9 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/acetonitrile/slurm_log/run_generation_acetonitrile_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC#N" -r 1613119 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/acetonitrile_starting_coordinates/ -sl Calculated_data/acetonitrile/ --oneonly'

# # # acetone			CC(=O)C			58.08	0.7845
# # mkdir /cluster/project/igc/kpaul/acetone_starting_coordinates
# # job10=$(sbatch --dependency=afterany:$job9 --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_acetone_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(=O)C" -r 16131120 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/acetone_starting_coordinates/')
# sbatch --dependency=afterany:$job10 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/acetone/slurm_log/run_generation_acetone_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(=O)C" -r 16131110 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/acetone_starting_coordinates/ -sl Calculated_data/acetone/ --oneonly'
# sbatch --dependency=afterany:$job10 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/acetone/slurm_log/run_generation_acetone_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(=O)C" -r 16131110 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/acetone_starting_coordinates/ -sl Calculated_data/acetone/ --oneonly'

# aceticacid		CC(O)=O			60.052	1.049
# # mkdir /cluster/project/igc/kpaul/aceticacid_starting_coordinates
# # job11=$(sbatch --parsable --array=0-7411%500 -n 4 --time=4:01:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_aceticacid_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(O)=O" -r 16131121 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/aceticacid_starting_coordinates/')
# sbatch --dependency=afterany:$job11 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/aceticacid/slurm_log/run_generation_aceticacid_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(O)=O" -r 16131111 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/aceticacid_starting_coordinates/ -sl Calculated_data/aceticacid/ --oneonly'
# sbatch --dependency=afterany:$job11 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/aceticacid/slurm_log/run_generation_aceticacid_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(O)=O" -r 16131111 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/aceticacid_starting_coordinates/ -sl Calculated_data/aceticacid/ --oneonly'

# 14dioxane		O1CCOCC1 		88.106	1.033
# # mkdir /cluster/project/igc/kpaul/14dioxane_starting_coordinates
# # job12=$(sbatch --dependency=afterany:$job11 --parsable --array=0-7411%500 -n 4 --time=4:01:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_14dioxane_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O1CCOCC1" -r 16131122 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/14dioxane_starting_coordinates/')
# sbatch --dependency=afterany:$job12 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/14dioxane/slurm_log/run_generation_14dioxane_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O1CCOCC1" -r 16131112 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/14dioxane_starting_coordinates/ -sl Calculated_data/14dioxane/ --oneonly'
# sbatch --dependency=afterany:$job12 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/14dioxane/slurm_log/run_generation_14dioxane_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O1CCOCC1" -r 16131112 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/14dioxane_starting_coordinates/ -sl Calculated_data/14dioxane/ --oneonly'

# nitrobenzol		c1ccc(cc1)[N+](=O)[O-]	123.11	1.199
# # mkdir /cluster/project/igc/kpaul/nitrobenzol_starting_coordinates
# # job13=$(sbatch --dependency=afterany:$job12 --parsable --array=0-7411%500 -n 4 --time=4:01:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_nitrobenzol_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccc(cc1)[N+](=O)[O-]" -r 16131123 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/nitrobenzol_starting_coordinates/')
# sbatch --dependency=afterany:$job13 --array=0-5000 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/nitrobenzol/slurm_log/run_generation_nitrobenzol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccc(cc1)[N+](=O)[O-]" -r 16131113 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/nitrobenzol_starting_coordinates/ -sl Calculated_data/nitrobenzol/ --oneonly'
# sbatch --dependency=afterany:$job13 --array=5000-7411 -n 4 --time=4:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/nitrobenzol/slurm_log/run_generation_nitrobenzol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccc(cc1)[N+](=O)[O-]" -r 16131113 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/nitrobenzol_starting_coordinates/ -sl Calculated_data/nitrobenzol/ --oneonly'

# ### CHECK Random SEEDS ###
# # HMPA			O=P(N(C)C)(N(C)C)N(C)C	179.20	1.03
# mkdir /cluster/project/igc/kpaul/HMPA_starting_coordinates
# job14=$(sbatch --parsable --array=0-7411%500 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_HMPA_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=P(N(C)C)(N(C)C)N(C)C" -r 16131124 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/HMPA_starting_coordinates/')
# sbatch --dependency=afterany:$job14 --array=0-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/HMPA/slurm_log/run_generation_HMPA_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=P(N(C)C)(N(C)C)N(C)C" -r 16131114 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/HMPA_starting_coordinates/ -sl Calculated_data/HMPA/ --oneonly'

# # MTBE			O(C(C)(C)C)C		88.150	0.7404
# mkdir /cluster/project/igc/kpaul/MTBE_starting_coordinates
# job15=$(sbatch --dependency=afterany:$job14 --parsable --array=0-7411%500 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_MTBE_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O(C(C)(C)C)C" -r 16131125 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/MTBE_starting_coordinates/')
# sbatch --dependency=afterany:$job15 --array=0-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/MTBE/slurm_log/run_generation_MTBE_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O(C(C)(C)C)C" -r 16131115 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/MTBE_starting_coordinates/ -sl Calculated_data/MTBE/ --oneonly'

# # IPA			CC(O)C			60.096	0.786
# mkdir /cluster/project/igc/kpaul/IPA_starting_coordinates
# job16=$(sbatch --parsable --array=0-7411%500 -n 4 --time=4:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_IPA_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(O)C" -r 16131126 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/IPA_starting_coordinates/')
# sbatch --dependency=afterany:$job16 --array=0-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/IPA/slurm_log/run_generation_IPA_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(O)C" -r 16131116 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/IPA_starting_coordinates/ -sl Calculated_data/IPA/ --oneonly'

# # Propylenecarbonate	CC1COC(=O)O1		102.089	1.205
# mkdir /cluster/project/igc/kpaul/Propylenecarbonate_starting_coordinates
# job17=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Propylenecarbonate_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC1COC(=O)O1" -r 16131127 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Propylenecarbonate_starting_coordinates/')
# sbatch --dependency=afterany:$job17 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Propylenecarbonate/slurm_log/run_generation_Propylenecarbonate_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC1COC(=O)O1" -r 16131117 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Propylenecarbonate_starting_coordinates/ -sl Calculated_data/Propylenecarbonate/ --oneonly'

# # Hexafluorobenzene	Fc1c(F)c(F)c(F)c(F)c1F	186.056	1.6120
# mkdir /cluster/project/igc/kpaul/Hexafluorobenzene_starting_coordinates
# job18=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Hexafluorobenzene_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "Fc1c(F)c(F)c(F)c(F)c1F" -r 16131128 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Hexafluorobenzene_starting_coordinates/')
# sbatch --dependency=afterany:$job18 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Hexafluorobenzene/slurm_log/run_generation_Hexafluorobenzene_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "Fc1c(F)c(F)c(F)c(F)c1F" -r 16131118 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Hexafluorobenzene_starting_coordinates/ -sl Calculated_data/Hexafluorobenzene/ --oneonly'

# NOTE Seed mismatch above generation more important but runseed would ideally match

# # pyridine		c1ccncc1		79.102	0.9819
# mkdir /cluster/project/igc/kpaul/pyridine_starting_coordinates
# job19=$(sbatch --parsable --array=0-7411 -n 4 --time=24:01:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_pyridine_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccncc1" -r 16131129 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/pyridine_starting_coordinates/')
# sbatch --dependency=afterany:$job19 --array=0-7411 -n 4 --time=24:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/pyridine/slurm_log/run_generation_pyridine_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "c1ccncc1" -r 16131129 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/pyridine_starting_coordinates/ -sl Calculated_data/pyridine/ --oneonly'

# # THF			C1CCOC1			72.107	0.8876
# mkdir /cluster/project/igc/kpaul/THF_starting_coordinates
# job20=$(sbatch --parsable --array=0-7411 -n 4 --time=24:01:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_THF_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1CCOC1" -r 16131130 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/THF_starting_coordinates/')
# sbatch --dependency=afterany:$job20 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/THF/slurm_log/run_generation_THF_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1CCOC1" -r 16131130 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/THF_starting_coordinates/ -sl Calculated_data/THF/ --oneonly'

# # Ethylacetate		O=C(OCC)C		88.106	0.902
# mkdir /cluster/project/igc/kpaul/Ethylacetate_starting_coordinates
# job21=$(sbatch --parsable --array=0-7411 -n 4 --time=24:01:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Ethylacetate_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=C(OCC)C" -r 16131131 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Ethylacetate_starting_coordinates/')
# sbatch --dependency=afterany:$job21 --array=0-7411 -n 4 --time=24:01:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Ethylacetate/slurm_log/run_generation_Ethylacetate_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "O=C(OCC)C" -r 16131131 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Ethylacetate_starting_coordinates/ -sl Calculated_data/Ethylacetate/ --oneonly'

# # Sulfolane		C1CCS(=O)(=O)C1		120.17	1.261
# mkdir /cluster/project/igc/kpaul/Sulfolane_starting_coordinates
# job22=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Sulfolane_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1CCS(=O)(=O)C1" -r 16131132 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Sulfolane_starting_coordinates/')
# sbatch --dependency=afterany:$job22 --array=0-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Sulfolane/slurm_log/run_generation_Sulfolane_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1CCS(=O)(=O)C1" -r 16131132 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Sulfolane_starting_coordinates/ -sl Calculated_data/Sulfolane/ --oneonly'

# # nitromethane		C[N+](=O)[O-]		61.04	1.1371
# mkdir /cluster/project/igc/kpaul/nitromethane_starting_coordinates
# job23=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_nitromethane_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C[N+](=O)[O-]" -r 16131133 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/nitromethane_starting_coordinates/')
# sbatch --dependency=afterany:$job23 --array=0-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/nitromethane/slurm_log/run_generation_nitromethane_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C[N+](=O)[O-]" -r 16131133 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/nitromethane_starting_coordinates/ -sl Calculated_data/nitromethane/ --oneonly'

# # Butylformate		CC(C)(C)OC=O		102.133	0.872
# mkdir /cluster/project/igc/kpaul/Butylformate_starting_coordinates
# job24=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Butylformate_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(C)(C)OC=O" -r 16131134 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Butylformate_starting_coordinates/')
# sbatch --dependency=afterany:$job24 --array=0-7411 -n 4 --time=4:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Butylformate/slurm_log/run_generation_Butylformate_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(C)(C)OC=O" -r 16131134 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Butylformate_starting_coordinates/ -sl Calculated_data/Butylformate/ --oneonly'

# # NMP			CN1CCCC1=O		99.133	1.028
# mkdir /cluster/project/igc/kpaul/NMP_starting_coordinates
# job25=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_NMP_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CN1CCCC1=O" -r 16131135 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/NMP_starting_coordinates/')
# sbatch --dependency=afterany:$job25 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/NMP/slurm_log/run_generation_NMP_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CN1CCCC1=O" -r 16131135 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/NMP_starting_coordinates/ -sl Calculated_data/NMP/ --oneonly'

# # ethyllactate		CCOC(=O)C(C)O		118.132	1.03
# mkdir /cluster/project/igc/kpaul/ethyllactate_starting_coordinates
# job26=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_ethyl_lactate_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCOC(=O)C(C)O" -r 16131136 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/ethyllactate_starting_coordinates/')
# sbatch --dependency=afterany:$job26 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/ethyl_lactate/slurm_log/run_generation_ethyl_lactate_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCOC(=O)C(C)O" -r 16131136 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/ethyllactate_starting_coordinates/ -sl Calculated_data/ethyl_lactate/ --oneonly'

# # Octanol			CCCCCCCCO		130.231	0.83
# mkdir /cluster/project/igc/kpaul/Octanol_starting_coordinates
# job27=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_Octanol_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCCCCCCCO" -r 16131137 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Octanol_starting_coordinates/')
# sbatch --dependency=afterany:$job27 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/Octanol/slurm_log/run_generation_Octanol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCCCCCCCO" -r 16131137 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Octanol_starting_coordinates/ -sl Calculated_data/Octanol/ --oneonly'

# # cyclohexane		C1CCCCC1		84.162	0.7739
# mkdir /cluster/project/igc/kpaul/cyclohexane_starting_coordinates
# job28=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_cyclohexane_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1CCCCC1" -r 16131138 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/cyclohexane_starting_coordinates/')
# sbatch --dependency=afterany:$job28 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/cyclohexane/slurm_log/run_generation_cyclohexane_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1CCCCC1" -r 16131138 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/cyclohexane_starting_coordinates/ -sl Calculated_data/cyclohexane/ --oneonly'

# # glycerin		OCC(O)CO		92.094	1.261
# mkdir /cluster/project/igc/kpaul/glycerin_starting_coordinates
# job29=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=2000 -o Calculated_data/slurm_log/run_generation_glycerin_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "OCC(O)CO" -r 16131139 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/glycerin_starting_coordinates/')
# sbatch --dependency=afterany:$job29 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=2000 --gpus=1 -o Calculated_data/glycerin/slurm_log/run_generation_glycerin_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "OCC(O)CO" -r 16131139 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/glycerin_starting_coordinates/ -sl Calculated_data/glycerin/ --oneonly'

# # carbontetrachloride	ClC(Cl)(Cl)Cl 		153.81 	1.5867
# mkdir /cluster/project/igc/kpaul/carbontetrachloride_starting_coordinates
# job30=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_carbontetrachloride_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)(Cl)Cl" -r 16131140 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/carbontetrachloride_starting_coordinates/')
# sbatch --dependency=afterany:$job30 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/carbontetrachloride/slurm_log/run_generation_carbontetrachloride_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "ClC(Cl)(Cl)Cl" -r 16131140 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/carbontetrachloride_starting_coordinates/ -sl Calculated_data/carbontetrachloride/ --oneonly'

# # DME			COCCOC			90.122	0.8683
# mkdir /cluster/project/igc/kpaul/DME_starting_coordinates
# job31=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_DME_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "COCCOC" -r 16131141 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/DME_starting_coordinates/')
# sbatch --dependency=afterany:$job31 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/DME/slurm_log/run_generation_DME_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "COCCOC" -r 16131141 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/DME_starting_coordinates/ -sl Calculated_data/DME/ --oneonly'

# # 2Nitropropane		CC(C)[N+](=O)[O-]	89.094	0.9821
# mkdir /cluster/project/igc/kpaul/2Nitropropane_starting_coordinates
# job32=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_2Nitropropane_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(C)[N+](=O)[O-]" -r 16131142 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/2Nitropropane_starting_coordinates/')
# sbatch --dependency=afterany:$job32 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/2Nitropropane/slurm_log/run_generation_2Nitropropane_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC(C)[N+](=O)[O-]" -r 16131142 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/2Nitropropane_starting_coordinates/ -sl Calculated_data/2Nitropropane/ --oneonly'

# # Trifluorotoluene	C1=CC=C(C=C1)C(F)(F)F	146.11	1.19
# mkdir /cluster/project/igc/kpaul/Trifluorotoluene_starting_coordinates
# job33=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_Trifluorotoluene_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1=CC=C(C=C1)C(F)(F)F" -r 16131143 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Trifluorotoluene_starting_coordinates/')
# sbatch --dependency=afterany:$job33 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/Trifluorotoluene/slurm_log/run_generation_Trifluorotoluene_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "C1=CC=C(C=C1)C(F)(F)F" -r 16131143 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Trifluorotoluene_starting_coordinates/ -sl Calculated_data/Trifluorotoluene/ --oneonly'

# # hexafluroacetone	FC(F)(F)C(=O)C(F)(F)F	166.02	1.32
# mkdir /cluster/project/igc/kpaul/hexafluroacetone_starting_coordinates
# job34=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_hexafluroacetone_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "FC(F)(F)C(=O)C(F)(F)F" -r 16131144 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/hexafluroacetone_starting_coordinates/')
# sbatch --dependency=afterany:$job34 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/hexafluroacetone/slurm_log/run_generation_hexafluroacetone_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "FC(F)(F)C(=O)C(F)(F)F" -r 16131144 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/hexafluroacetone_starting_coordinates/ -sl Calculated_data/hexafluroacetone/ --oneonly'

# # Propionitrile		CCC#N			55.080	0.772
# mkdir /cluster/project/igc/kpaul/Propionitrile_starting_coordinates
# job35=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_Propionitrile_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCC#N" -r 16131145 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Propionitrile_starting_coordinates/')
# sbatch --dependency=afterany:$job35 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/Propionitrile/slurm_log/run_generation_Propionitrile_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CCC#N" -r 16131145 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Propionitrile_starting_coordinates/ -sl Calculated_data/Propionitrile/ --oneonly'

# # Benzonitrile		N#Cc1ccccc1		103.12	1.0
# mkdir /cluster/project/igc/kpaul/Benzonitrile_starting_coordinates
# job36=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_Benzonitrile_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "N#Cc1ccccc1" -r 16131146 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/Benzonitrile_starting_coordinates/')
# sbatch --dependency=afterany:$job36 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/Benzonitrile/slurm_log/run_generation_Benzonitrile_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "N#Cc1ccccc1" -r 16131146 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/Benzonitrile_starting_coordinates/ -sl Calculated_data/Benzonitrile/ --oneonly'

# # oxylol			CC1=C(C)C=CC=C1 	106.168	0.88
# mkdir /cluster/project/igc/kpaul/oxylol_starting_coordinates
# job37=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000  --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_oxylol_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC1=C(C)C=CC=C1" -r 16131147 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/oxylol_starting_coordinates/')
# sbatch --dependency=afterany:$job37 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000  --mem-per-cpu=4000 --gpus=1 -o Calculated_data/oxylol/slurm_log/run_generation_oxylol_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "CC1=C(C)C=CC=C1" -r 16131147 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/oxylol_starting_coordinates/ -sl Calculated_data/oxylol/ --oneonly'

# TIP5P O
# mkdir /cluster/project/igc/kpaul/TIP5P_O_starting_coordinates
# job38=$(sbatch --parsable --array=0-7411 -n 4 --time=24:00:00 --tmp=1000 --mem-per-cpu=4000 -o Calculated_data/slurm_log/run_generation_TIP5P_O_starting_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "TIP5P" -r 16131148 -f generation_smiles/canonical_smiles.npy --startonly --starttrajloc /cluster/project/igc/kpaul/TIP5P_O_starting_coordinates/')
# sbatch --dependency=afterany:$job38 --array=0-7411 -n 4 --time=24:00:00 --tmp=5000 --mem-per-cpu=4000 --gpus=1 -o Calculated_data/TIP5P_O/slurm_log/run_generation_TIP5P_O_%A_%a.out --wrap='python run_training_set_generation.py -i $SLURM_ARRAY_TASK_ID -n 50 -s "TIP5P" -r 16131148 -f generation_smiles/canonical_smiles.npy -nf 1 --starttrajloc /cluster/project/igc/kpaul/TIP5P_O_starting_coordinates/ -sl Calculated_data/TIP5P_O/ --oneonly'

