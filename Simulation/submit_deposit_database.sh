# This file is used to submit the database deposition
### WARNING DO NOT SUBMIT MORE THAN ONE JOB AT A TIME OR YOU WILL REACH THE NUMBER OF CONNECTION LIMITS OF THE DATABASE ###
### NOTE: 17.06.2024: Some files have been moved to the Solvent Folders
# sbatch -n 32 --time=24:00:00  --mem-per-cpu=2000 -o simulation/slurm_log/run_sim_%A_%a.out --wrap='python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/ClC(Cl)Cl_small_molecules_n_50_id_XXX_seed_161311.hdf5" -nc 32 -d solvent_chloroform'
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/CO_small_molecules_n_50_id_XXX_seed_161311.hdf5" -nc 32 -d solvent_methanol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/CS(=O)C_small_molecules_n_50_id_XXX_seed_161311.hdf5" -nc 32 -d solvent_dmso
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Octanol/CCCCCCCCO_small_molecules_n_50_id_XXX_seed_16131137.hdf5" -nc 32 -d solvent_octanol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/DMPU/O=C1N(C)CCCN1C_small_molecules_n_50_id_XXX_seed_1613111.hdf5" -nc 32 -d solvent_dmpu
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Diethylether/CCOCC_small_molecules_n_50_id_XXX_seed_1613112.hdf5" -nc 32 -d solvent_diethylether
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Ethanol/OCC_small_molecules_n_50_id_XXX_seed_1613113.hdf5" -nc 32 -d solvent_ethanol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/DMF/CN(C)C=O_small_molecules_n_50_id_XXX_seed_1613114.hdf5" -nc 32 -d solvent_dmf
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/DCM/ClCCl_small_molecules_n_50_id_XXX_seed_1613115.hdf5" -nc 32 -d solvent_dcm
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Toluol/Cc1ccccc1_small_molecules_n_50_id_XXX_seed_1613116.hdf5" -nc 32 -d solvent_toluol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Benzol/c1ccccc1_small_molecules_n_50_id_XXX_seed_1613117.hdf5" -nc 32 -d solvent_benzol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Hexan/CCCCCC_small_molecules_n_50_id_XXX_seed_1613118.hdf5" -nc 32 -d solvent_hexan
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/acetonitrile/CC#N_small_molecules_n_50_id_XXX_seed_1613119.hdf5" -nc 32 -d solvent_acetonitrile
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/acetone/CC(=O)C_small_molecules_n_50_id_XXX_seed_16131110.hdf5" -nc 32 -d solvent_acetone
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/aceticacid/CC(O)=O_small_molecules_n_50_id_XXX_seed_16131111.hdf5" -nc 32 -d solvent_aceticacid
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/14dioxane/O1CCOCC1_small_molecules_n_50_id_XXX_seed_16131112.hdf5" -nc 32 -d solvent_14dioxane
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/nitrobenzol/c1ccc(cc1)[N+](=O)[O-]_small_molecules_n_50_id_XXX_seed_16131113.hdf5" -nc 32 -d solvent_nitrobenzol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/HMPA/O=P(N(C)C)(N(C)C)N(C)C_small_molecules_n_50_id_XXX_seed_16131114.hdf5" -nc 32 -d solvent_hmpa
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/MTBE/O(C(C)(C)C)C_small_molecules_n_50_id_XXX_seed_16131115.hdf5" -nc 32 -d solvent_mtbe
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/IPA/CC(O)C_small_molecules_n_50_id_XXX_seed_16131116.hdf5" -nc 32 -d solvent_ipa
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Propylenecarbonate/CC1COC(=O)O1_small_molecules_n_50_id_XXX_seed_16131117.hdf5" -nc 32 -d solvent_propylenecarbonate
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Hexafluorobenzene/Fc1c(F)c(F)c(F)c(F)c1F_small_molecules_n_50_id_XXX_seed_16131118.hdf5" -nc 32 -d solvent_hexafluorobenzene
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/pyridine/c1ccncc1_small_molecules_n_50_id_XXX_seed_16131129.hdf5" -nc 32 -d solvent_pyridine
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/THF/C1CCOC1_small_molecules_n_50_id_XXX_seed_16131130.hdf5" -nc 32 -d solvent_thf
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Ethylacetate/O=C(OCC)C_small_molecules_n_50_id_XXX_seed_16131131.hdf5" -nc 32 -d solvent_ethylacetate
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Sulfolane/C1CCS(=O)(=O)C1_small_molecules_n_50_id_XXX_seed_16131132.hdf5" -nc 32 -d solvent_sulfolane
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/nitromethane/C[N+](=O)[O-]_small_molecules_n_50_id_XXX_seed_16131133.hdf5" -nc 32 -d solvent_nitromethane
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Butylformate/CC(C)(C)OC=O_small_molecules_n_50_id_XXX_seed_16131134.hdf5" -nc 32 -d solvent_butylformate
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/NMP/CN1CCCC1=O_small_molecules_n_50_id_XXX_seed_16131135.hdf5" -nc 32 -d solvent_nmp
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/ethyllactate/CCOC(=O)C(C)O_small_molecules_n_50_id_XXX_seed_16131136.hdf5" -nc 32 -d solvent_ethyllactate
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Octanol/CCCCCCCCO_small_molecules_n_50_id_XXX_seed_16131137.hdf5" -nc 32 -d solvent_octanol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/cyclohexane/C1CCCCC1_small_molecules_n_50_id_XXX_seed_16131138.hdf5" -nc 32 -d solvent_cyclohexane
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/glycerin/OCC(O)CO_small_molecules_n_50_id_XXX_seed_16131139.hdf5" -nc 32 -d solvent_glycerin
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/carbontetrachloride/ClC(Cl)(Cl)Cl_small_molecules_n_50_id_XXX_seed_16131140.hdf5" -nc 32 -d solvent_carbontetrachloride
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/DME/COCCOC_small_molecules_n_50_id_XXX_seed_16131141.hdf5" -nc 32 -d solvent_dme
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/2Nitropropane/CC(C)[N+](=O)[O-]_small_molecules_n_50_id_XXX_seed_16131142.hdf5" -nc 32 -d solvent_2nitropropane
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Trifluorotoluene/C1=CC=C(C=C1)C(F)(F)F_small_molecules_n_50_id_XXX_seed_16131143.hdf5" -nc 32 -d solvent_trifluorotoluene
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/hexafluroacetone/FC(F)(F)C(=O)C(F)(F)F_small_molecules_n_50_id_XXX_seed_16131144.hdf5" -nc 32 -d solvent_hexafluroacetone
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Propionitrile/CCC#N_small_molecules_n_50_id_XXX_seed_16131145.hdf5" -nc 32 -d solvent_propionitrile
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/Benzonitrile/N#Cc1ccccc1_small_molecules_n_50_id_XXX_seed_16131146.hdf5" -nc 32 -d solvent_benzonitrile
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/oxylol/CC1=C(C)C=CC=C1_small_molecules_n_50_id_XXX_seed_16131147.hdf5" -nc 32 -d solvent_oxylol
python run_deposit_database.py -s 0 -e 7412 -t train -h5 "Calculated_data/TIP5P_O/TIP5P_small_molecules_n_50_id_XXX_seed_16131147.hdf5" -nc 32 -d solvent_tip5p_all

# TIP5P O

