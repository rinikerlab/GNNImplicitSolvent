# Run locally 20.06.2024 with:
# python collect_complete_dataset.py -n 20 -pt /media/kpaul/173c6995-ac7e-467a-9881-4c8910d98285/kpaul/small_molecule_multisolvent/MachineLearning/datasets/FullTrainingSet/full_ds_XXX.pt
# for collecting the test set:
# python collect_complete_dataset.py -u test -n 1 -pt test.pt

import sys

sys.path.append("../")
import psycopg2
from GNN_Trainer import Trainer
import torch
import yaml
import tqdm
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Run dataset_generation")
parser.add_argument("-pt", "--ptfile", type=str, help="path to pt file")
parser.add_argument("-n", "--number_of_files", type=int, help="number of files")
parser.add_argument("-u", "--usage_flag", type=str, default="train", help="usage flag")
args = parser.parse_args()

solvent_dict = yaml.load(open("../Simulation/solvents.yml"), Loader=yaml.FullLoader)[
    "solvent_mapping_dict"
]

with open("/home/kpaul/.lwreg_key", "r") as f:
    key = f.read().strip()

config = {}
config["dbname"] = "solvent_forces"
config["dbtype"] = "postgresql"
config["removeHs"] = 0
config["registerConformers"] = True
config["hashConformer"] = 0  # set to 0
config["numConformerDigits"] = 3  # Question: what is this?
config["host"] = "scotland"
config["user"] = "kpaul_lwreg"
config["password"] = key

# Get connection
cn = psycopg2.connect(
    database=config.get("dbname", None),
    host=config.get("host", None),
    user=config.get("user", None),
    password=config.get("password", None),
)
curs = cn.cursor()

# Fix seed
np.random.seed(161311)

# Get all the conf_ids
conf_id_arrays = {}
# Select solvent
for solvent in tqdm.tqdm(solvent_dict.keys()):
    curs.execute(
        "select conf_id from solvent_%s.explicit_calculations where usage_flag='%s'"
        % (solvent, args.usage_flag)
    )
    conf_id_array = np.array([ci[0] for ci in curs.fetchall()], dtype=int)
    conf_id_array = np.sort(conf_id_array)
    conf_id_array = np.random.permutation(conf_id_array)
    conf_id_arrays[solvent] = conf_id_array
    print("Dataset size: ", solvent, conf_id_array.size)

number_of_files = args.number_of_files
ptfile = args.ptfile

for i in tqdm.tqdm(range(number_of_files)):
    print("Working on file %i" % i)
    if os.path.exists(ptfile.replace("XXX", "%i" % i)):
        continue
    full_ds = []
    for solvent in solvent_dict.keys():
        solvent_id = solvent_dict[solvent]["solvent_id"]
        solvent_dielectric = solvent_dict[solvent]["dielectric"]
        shuffeled_conf_id_array = conf_id_arrays[solvent]
        number_of_parallel_entries = int(
            np.ceil(shuffeled_conf_id_array.size / number_of_files)
        )
        print("Individual chunks: ", solvent, solvent_id, number_of_parallel_entries)
        curs.execute(
            "select positions,forces,atomfeatures from solvent_"
            + solvent
            + ".explicit_calculations where conf_id in %s",
            (
                tuple(
                    shuffeled_conf_id_array[
                        number_of_parallel_entries
                        * i : number_of_parallel_entries
                        * (i + 1)
                    ].tolist()
                ),
            ),
        )
        ds = Trainer.fetch_training_data_from_query_pairs(
            curs, solvent_id=solvent_id, solvent_dielectric=solvent_dielectric
        )
        print("Extracted data: ", len(ds))
        full_ds += ds
    print("Total size: ", len(full_ds))
    filename = ptfile.replace("XXX", "%i" % i)
    torch.save(full_ds, filename)

cn.close()
