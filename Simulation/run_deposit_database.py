import argparse

parser = argparse.ArgumentParser(description="Run deposit database")
parser.add_argument("-s", "--start", type=int, help="Startindex")
parser.add_argument("-e", "--end", type=int, help="Endindex")
parser.add_argument("-t", "--type", type=str, help="type for deposition")
parser.add_argument("-h5", "--hdf5", type=str, help="hdf5 file core")
parser.add_argument("-nc", "--num_cores", type=int, help="number of cores", default=1)
parser.add_argument("-d", "--database", type=str, help="database to deposit to")
parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
args = parser.parse_args()

import lwreg
from lwreg.utils import standardization_lib
from rdkit import Chem
import numpy as np
import sys

sys.path.append("..")

from Data.Datahandler import hdf5_storage
from rdkit.Geometry import Point3D
from multiprocessing import Pool
import tqdm
import os

if os.path.isfile("/cluster/home/kpaul/.lwreg_key"):
    with open("/cluster/home/kpaul/.lwreg_key", "r") as f:
        key = f.read().strip()
else:
    with open("/home/kpaul/.lwreg_key", "r") as f:
        key = f.read().strip()

config = lwreg.utils.defaultConfig()
config["standardization"] = standardization_lib.NoStandardization()
config["dbname"] = "solvent_forces"
config["dbtype"] = "postgresql"
config["removeHs"] = 0
config["registerConformers"] = True
config["hashConformer"] = 0  # set to 0
config["numConformerDigits"] = 3  # Question: what is this?
config["host"] = "scotland"
config["user"] = "kpaul_lwreg"
config["password"] = key


def create_table(data_base):
    cn = lwreg.utils.connect(config)  # Connection to the database
    curs = cn.cursor()  # Command line cursor in postgresql
    curs.execute("create schema if not exists %s" % data_base)  # execute the command
    curs.execute(
        "create table if not exists %s.explicit_calculations (conf_id int primary key, positions float[][] not null, forces float[][] not null, atomfeatures float[][] not null, trajectory text, origin text, usage_flag text not null)"
        % data_base
    )  # execute the command
    cn.commit()  # commit the command


create_table(args.database)


def arr_to_string(arr):
    arraystring = np.array2string(arr, separator=",")
    arraystring = arraystring.replace("[", "{")
    arraystring = arraystring.replace("]", "}")
    return arraystring


# Completely rewritten deposit_file function
def deposit_file(file, usage_flag=args.type, data_base=args.database):

    deposited_entries = 0

    if not os.path.isfile(file):
        print("file does not exist")
        print(file)
        return -1
    storage = hdf5_storage(file)

    try:
        mcdict = storage.get_molids_and_confids()
    except:
        return -1

    cn = lwreg.utils.connect(config)  # Connection to the database
    curs = cn.cursor()  # Command line cursor in postgresql

    for key in mcdict.keys():

        try:
            confid = mcdict[key][0]
            smiles = storage.get_smiles(key)
        except Exception as e:
            if args.verbose:
                print(e)
            continue
        try:
            force, pos, frame = storage.get_reextraction(key, confid, True)
            atomfeatures = storage.get_repocessed_atom_features_and_unique_radii(
                key, confid
            )
            atfeat = atomfeatures[0]
        except Exception as e:
            if args.verbose:
                print(e)
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            conf = mol.GetConformer()
        except Exception as e:
            if args.verbose:
                print(e)
            continue

        for k in range(len(pos)):
            try:
                for i in range(mol.GetNumAtoms()):
                    x, y, z = pos[k][i]
                    conf.SetAtomPosition(i, Point3D(x, y, z))
            except Exception as e:
                if args.verbose:
                    print(e)
                continue
            try:
                entry_file = file.split("/")[-1]
                entry_molregno_confid = lwreg.register(
                    config, mol, fail_on_duplicate=True, no_verbose=True
                )
                curs.execute(
                    "insert into "
                    + args.database
                    + ".explicit_calculations (conf_id, positions, forces, atomfeatures, trajectory, origin, usage_flag) values (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        entry_molregno_confid[1],
                        pos[k].tolist(),
                        force[k].tolist(),
                        atfeat.tolist(),
                        None,
                        entry_file,
                        usage_flag,
                    ),
                )
                cn.commit()
                deposited_entries += 1
            except Exception as e:
                if args.verbose:
                    print(e)
                try:
                    entry_file = file.split("/")[-1]
                    # print(entry_file)
                    entry_molregno_confid = lwreg.register(
                        config, mol, fail_on_duplicate=False, no_verbose=True
                    )
                    curs.execute(
                        "select exists(select 1 from "
                        + args.database
                        + ".explicit_calculations  where conf_id = %i)"
                        % entry_molregno_confid[1]
                    )
                    exists = curs.fetchone()[0]

                    if not exists:
                        curs.execute(
                            "insert into "
                            + args.database
                            + ".explicit_calculations (conf_id, positions, forces, atomfeatures, trajectory, origin, usage_flag) values (%s, %s, %s, %s, %s, %s, %s)",
                            (
                                entry_molregno_confid[1],
                                pos[k].tolist(),
                                force[k].tolist(),
                                atfeat.tolist(),
                                None,
                                entry_file,
                                usage_flag,
                            ),
                        )
                        cn.commit()
                        deposited_entries += 1
                except Exception as e:
                    # print(k)
                    if args.verbose:
                        print(e)
                    continue
    if args.verbose:
        print(file, deposited_entries)
    if "100.h5" in file:
        print(file, deposited_entries)
    cn.commit()
    return deposited_entries


from copy import deepcopy


def get_name_of_file(file, id):

    wfile = deepcopy(file)
    wfile = wfile.replace("XXX", str(id))
    return wfile


files = [get_name_of_file(args.hdf5, i) for i in range(args.start, args.end)]

if args.num_cores > 1:
    pool = Pool(args.num_cores)
    returns = pool.map(deposit_file, files)
else:
    returns = []
    for file in tqdm.tqdm(files):
        returns.append(deposit_file(file))
