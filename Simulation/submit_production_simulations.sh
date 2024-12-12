# Intramolecular H-Bond
sbatch --array=0,1,2 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/production_run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 10 -a all_solvent_run_seed_1612 -mfile ../MachineLearning/trained_models/ProductionRun_seed_1612_49_ckpt.pt -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol'

for seed in 1613111 1613112 1613113
do
for solvent in TIP5P tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol
do
sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r '$seed''
done
done

# Dimethoxy
sbatch --array=0 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/production_run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/dimethoxy.txt -nr 10 -a all_solvent_run_seed_1612 -mfile ../MachineLearning/trained_models/ProductionRun_seed_1612_49_ckpt.pt -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol'
