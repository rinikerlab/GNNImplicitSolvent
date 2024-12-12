import sys

sys.path.append("../")
from GNN_Trainer import Trainer, FilebasedDataset
from GNN_Models import GNN3_Multisolvent_embedding
from GNN_Loss_Functions import calculate_force_loss_only
import yaml
import torch
import time
import numpy as np
from typing import DefaultDict
from torch_geometric.loader import DataLoader
import argparse
import wandb

parser = argparse.ArgumentParser(description="Run Model Training")
parser.add_argument("-c", "--config", type=str, help="config file")
parser.add_argument("-s", "--seed", type=int, help="seed", default=-1)
parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
args = parser.parse_args()

with open(args.config, "r") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

if args.seed != -1:
    params["seed"] = args.seed
    params["name"] = params["name"] + "_seed_" + str(args.seed)

solvent_dict = yaml.load(
    open("../Simulation/solvents.yml", "r"), Loader=yaml.FullLoader
)["solvent_mapping_dict"]

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.manual_seed(params["seed"])

wandb.init(project="GNN Implicit Solvent", config=params, name=params["name"])
trainer = Trainer(
    verbose=args.verbose,
    name=params["name"],
    path="trained_models",
    force_mode=True,
    enable_tmp_dir=False,
    random_state=params["random"],
)

device = "cuda"
trainer.explicit = True
gbneck_parameters = None
unique_radii = [0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13]
model_class = GNN3_Multisolvent_embedding
model = model_class(
    radius=params["radius"],
    max_num_neighbors=10000,
    parameters=gbneck_parameters,
    device=device,
    fraction=params["fra"],
    unique_radii=unique_radii,
    num_solvents=params["num_solvents"],
    hidden=params["hidden"],
    dropout_rate=params["dropout"],
    hidden_token=params["hidden_token"],
    scaling_factor=params["scaling_factor"],
)

trainer.model = model
trainer.initialize_optimizer(params["lr"], params["scheduler"])
trainer.set_lossfunction(calculate_force_loss_only)

solvent_name_dict = DefaultDict(lambda: "Unknown Solvent")
for solvent in solvent_dict.keys():
    solvent_name_dict[solvent_dict[solvent]["solvent_id"]] = solvent

for epoch in range(params["epochs"]):
    dataset = FilebasedDataset(params["ptfolder"], params["use_tmpdir"])
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        prefetch_factor=params["dataloader_prefetch_factor"],
        num_workers=params["dataloader_num_workers"],
    )
    start = time.time()
    for f, full_ds in enumerate(dataloader):
        wandb.log({"loading_time": time.time() - start, "epoch": epoch, "step": f})
        sptime = time.time()
        trainer.create_splitted_data(full_ds, params["per"])
        wandb.log({"splitting_time": time.time() - sptime, "epoch": epoch, "step": f})
        trainer.train_model(
            1,
            params["batchsize"],
            params["clip"],
            save_model_after_epochs=False,
            wandblogging=True,
            make_scheduler_step=False,
        )

        if f % params["logging_frequency"] == 0:
            solvent_loss = trainer.validate_per_solvent(
                params["batchsize"],
                log_graph=True,
                solvent_name_dict=solvent_name_dict,
            )
            logdict = {"epoch": epoch, "step": f}
            for i, solv in enumerate(solvent_loss):
                logdict["solvent %s RMSE" % solvent_name_dict[i]] = np.sqrt(solv)
            wandb.log(logdict)
        start = time.time()

    trainer._scheduler.step()

    checkpoint = {
        "model": trainer._model.state_dict(),
        "optimizer": trainer._optimizer.state_dict(),
    }
    savename = "trained_models/" + params["name"] + "_" + str(epoch) + "_ckpt.pt"
    torch.save(checkpoint, savename)
