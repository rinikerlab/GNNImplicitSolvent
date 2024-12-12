# # Long TIP3P simulations
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/conformational_ensemble_smiles.txt -s "ClC(Cl)Cl" -r 161311'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/conformational_ensemble_smiles.txt -s "CO" -r 161311'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/conformational_ensemble_smiles.txt -s "CS(=O)C" -r 161311'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/conformational_ensemble_smiles.txt -s "CCCCCCCCO" -r 161311'

# # intramolecular H-Bond
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s O -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "ClC(Cl)Cl" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "CO" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "CS(=O)C" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "CCCCCCCCO" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "O=P(N(C)C)(N(C)C)N(C)C" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "O=C(OCC)C" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "OCC(O)CO" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "FC(F)(F)C(=O)C(F)(F)F" -r 161311'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 200 -f simulation_smiles/intra_molecular_hbond2.txt -s "c1ccc(cc1)[N+](=O)[O-]" -r 161311'

# for solvent in Chloroform IPA Methanol
# do
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r 161311'
# done
# for solvent in Chloroform IPA Methanol

# for solvent in TIP5P tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol
# do
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r 161311'
# done

# # Run NMR molecule
# for solvent in TIP5P tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol
# do
# sbatch --array=0 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_dimethoxy_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/dimethoxy.txt -s "'$solvent'" -r 161311'
# done

# # Run 2 Balances
# for solvent in Chloroform Methanol DMSO Ethanol DCM acetonitrile acetone Ethylacetate THF
# do
# sbatch --array=6,16 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_MB_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/MB.txt -s "'$solvent'" -r 161311'
# done


# Run additional seeds
# for seed in 1613111 1613112 1613113
# do
# for solvent in TIP5P tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol
# do
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r '$seed''
# done
# done

# sbatch --array=0,1,2 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 10 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 10 -a all_solvent_run_LargerBornscalingmodel -f 0.05 -sf 4.0 -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/LargerBornscalingmodel.model'
# sbatch --array=0,1,2 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 10 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 50 -a all_solvent_run_LargerBornscalingmodel_long -f 0.05 -sf 4.0 -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/LargerBornscalingmodel.model'
# sbatch --array=0,1,2 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 10 -a all_solvent_run_LargerBornscalingmodel_long -f 0.05 -sf 4.0 -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/LargerBornscalingmodel.model'


# sbatch --array=0,1,2 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 10 -a all_solvent_run -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/AdamWmodel.model'

# sbatch --array=0 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_dimethoxy_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/dimethoxy.txt -nr 10 -a all_solvent_run -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/AdamWmodel.model'

# Run 7-membered ring for longer
# for solvent in tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol
# do
# sbatch --array=2 -n 8 --gpus=1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 2500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r 161311'
# done

# Submit TIP3P TIP5P simulations
# for solvent in TIP5P
# do
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 2500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r 161311'
# done

# ExperimentalReference
# for solvent in TIP5P tip3p
# do
# sbatch --array=0,1 -n 8 --gpus=1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 2500 -f simulation_smiles/expref.txt -s "'$solvent'" -r 161311'
# done

# sbatch --array=0,1 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/expref.txt -nr 10 -a all_solvent_run -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/AdamWmodel.model'

# for solvent in Chloroform IPA Methanol
# do
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s "'$solvent'" -r 161311'
# done


# for solvent in Chloroform IPA Methanol Ethylacetate DMSO THF HMPA
# do
# sbatch --array=0-14 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond.txt -s "'$solvent'" -r 161311'
# done

# sbatch --array=0,1,2 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 10 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 32 -a testother -sn tip3p tip5p_all Chloroform DMSO Methanol HMPA Ethylacetate hexafluroacetone glycerin IPA MTBE THF DME NMP -mfile ../MachineLearning/trained_models/AdamWmodel.model'
# sbatch --array=0-14 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 10 -fi simulation_smiles/intra_molecular_hbond.txt -nr 32 -a intragnnmultinr32 -sn tip3p tip5p_all Chloroform DMSO Methanol HMPA Ethylacetate hexafluroacetone glycerin IPA MTBE THF DME NMP -mfile ../MachineLearning/trained_models/AdamWmodel.model'


# GBNeck2 simulations
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/sim_intra2_gbneck_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s GBNeck2 -r 161311 -sd 4.81 -i 100'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/sim_intra2_gbneck_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s GBNeck2 -r 161311 -sd 6.2 -i 100'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/sim_intra2_gbneck_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s GBNeck2 -r 161311 -sd 35.6 -i 100'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/sim_intra2_gbneck_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s GBNeck2 -r 161311 -sd 47.24 -i 100'
# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/sim_intra2_gbneck_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond2.txt -s GBNeck2 -r 161311 -sd 78.5 -i 100'

# sbatch --array=0,1,2 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_gnn_sim_intra2_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 20 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 20 -sm 0 1 2 3 4 -sd 78.5 4.81 46.7 32.6 10.2 -b 64 -p 0.95 -f 0.1 -r 3 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -s 111 -mfile ../MachineLearning/trained_models/GNN3_pub__batchsize_64_per_0.95_fra_0.1_random_3_radius_0.6_lr_0.0005_epochs_30_modelid_64_name__clip_1.0_solvent_dielectric_111.0_verbose_False_limit_0model.model'
# # Conformational Ensemble
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_gnn_sim_conf_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 20 -fi simulation_smiles/conformational_ensemble_smiles.txt -nr 20 -sm 0 1 2 3 4 -sd 78.5 4.81 46.7 32.6 10.2 -b 64 -p 0.95 -f 0.1 -r 3 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -s 111 -mfile ../MachineLearning/trained_models/GNN3_pub__batchsize_64_per_0.95_fra_0.1_random_3_radius_0.6_lr_0.0005_epochs_30_modelid_64_name__clip_1.0_solvent_dielectric_111.0_verbose_False_limit_0model.model'

# # Long GBNeck2 simulations
# sbatch --array=0,1,2,3,4 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/conformational_ensemble_smiles.txt -s GBNeck2 -r 161311'

# # 5 chunks of 50ns simulations
# sbatch --array=0,1,2,3,4 -n 8 --gpus=1 --time=4:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 50 -f simulation_smiles/conformational_ensemble_smiles.txt -s O -r 1 -i 100'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=1 --time=4:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 50 -f simulation_smiles/conformational_ensemble_smiles.txt -s O -r 2 -i 100'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=1 --time=4:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 50 -f simulation_smiles/conformational_ensemble_smiles.txt -s O -r 3 -i 100'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=1 --time=4:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 50 -f simulation_smiles/conformational_ensemble_smiles.txt -s O -r 4 -i 100'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=1 --time=4:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 50 -f simulation_smiles/conformational_ensemble_smiles.txt -s O -r 5 -i 100'

# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_gnn_sim_%A_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 2 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 10 --file simulation_smiles/conformational_ensemble_smiles.txt -nr 128'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_gnn_sim_r1_%A_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 1 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 10 --file simulation_smiles/conformational_ensemble_smiles.txt -nr 128'
# sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_3090:1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_gnn_sim_r3_%A_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 3 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 10 --file simulation_smiles/conformational_ensemble_smiles.txt -nr 128'

# Intramolecular Hydrogen Bonds

#sbatch --array=0-14 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_tip3p_intra_molecular_hbond_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond.txt -s O -r 161311'
#sbatch --array=[0-14]%4 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gbneck_intra_molecular_hbond_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/intra_molecular_hbond.txt -s GBNeck2 -r 161311'
#sbatch --array=[0-14]%7 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gnn_1intra_molecular_hbond_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 1 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 5 --file simulation_smiles/intra_molecular_hbond.txt -nr 128'
#sbatch --array=[0-14]%5 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gnn_2intra_molecular_hbond_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 2 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 5 --file simulation_smiles/intra_molecular_hbond.txt -nr 128'
#sbatch --array=[0-14]%5 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gnn_3intra_molecular_hbond_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 3 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 5 --file simulation_smiles/intra_molecular_hbond.txt -nr 128'

# SAMPL4 Challenge
# sbatch --dependency=afterany:35268817 --array=[0-46]%5 -n 4 --gpus=1 --time=4:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_vac_sampl4_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 100 -f simulation_smiles/SAMPL4.txt -s vac -r 161311'

# sbatch --dependency=afterany:35268817 --array=[0-46]%5 -n 4 --gpus=1 --time=4:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gnn_1_sampl4_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 1 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 1 --file simulation_smiles/SAMPL4.txt -nr 128'
# sbatch --dependency=afterany:35268817 --array=[0-46]%5 -n 4 --gpus=1 --time=4:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gnn_2_sampl4_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 2 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 1 --file simulation_smiles/SAMPL4.txt -nr 128'
# sbatch --dependency=afterany:35268817 --array=[0-46]%5 -n 4 --gpus=1 --time=4:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gnn_3_sampl4_%a.out --wrap='python run_gnn_simulation.py -b 32 -p 0.95 -f 0.1 -r 3 -ra 0.6 -l 0.0005 -e 30 -m 64 -c 1.0 -id $SLURM_ARRAY_TASK_ID --ns 1 --file simulation_smiles/SAMPL4.txt -nr 128'

# sbatch --dependency=afterany:35268817 --array=[0-46]%5 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gbneck_sampl4_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/SAMPL4.txt -s GBNeck2 -r 161311'
# sbatch --array=[0-46]%25 -n 4 --gpus=1 --time=24:00:00  --mem-per-cpu=4000 -o simulation/slurm_log/run_gbneck_sampl4_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 100 -f simulation_smiles/SAMPL4.txt -s SAGBNeck2 -r 161311'


# ## Submit New Model implicit Simulations
# sbatch --array=0 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_dimethoxy_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/dimethoxy.txt -nr 10 -a all_solvent_run_ProductionRun -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/ProductionRunmodel.model'
# sbatch --array=0,1,2 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_intramolecular_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/intra_molecular_hbond2.txt -nr 10 -a all_solvent_run_ProductionRun -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/ProductionRunmodel.model'

# for solvent in tip3p Chloroform Methanol DMSO Toluol Hexan nitrobenzol acetonitrile THF
# do
# sbatch --array=0-59 -n 8 --gpus=1 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/platinum_%A_%a.out --wrap='python run_simulation_for_small_molecules.py -id $SLURM_ARRAY_TASK_ID -n 500 -f simulation_smiles/platinum_diverse_selection.txt -s "'$solvent'" -r 161311'
# done

# sbatch --array=0-29 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/platinum_GNN_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/platinum_diverse_selection.txt -nr 10 -a selected_solvent_run_ProductionRun -sn tip3p Chloroform Methanol DMSO Toluol Hexan nitrobenzol acetonitrile THF -mfile ../MachineLearning/trained_models/ProductionRunmodel.model'
# sbatch --array=30-59 -n 8 --gpus=rtx_3090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/platinum_GNN_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/platinum_diverse_selection.txt -nr 10 -a selected_solvent_run_ProductionRun -sn tip3p Chloroform Methanol DMSO Toluol Hexan nitrobenzol acetonitrile THF -mfile ../MachineLearning/trained_models/ProductionRunmodel.model'


sbatch --array=0,1,2,3,4 -n 8 --gpus=rtx_4090:1 --time=120:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_conformational_ensemble_%A_%a.out --wrap='python run_gnn_multisolvent_simulation.py -id $SLURM_ARRAY_TASK_ID -ns 50 -fi simulation_smiles/conformational_ensemble_smiles.txt -nr 10 -a all_solvent_run_ProductionRun -sn tip3p Chloroform Methanol DMSO DMPU Diethylether Ethanol DMF DCM Toluol Benzol Hexan acetonitrile acetone aceticacid 14dioxane nitrobenzol HMPA MTBE IPA Hexafluorobenzene pyridine THF Ethylacetate Sulfolane nitromethane Butylformate Octanol cyclohexane glycerin carbontetrachloride DME 2Nitropropane Trifluorotoluene hexafluroacetone Propionitrile Benzonitrile oxylol tip5p_all -mfile ../MachineLearning/trained_models/ProductionRunmodel.model'
