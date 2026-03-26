import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryAveragePrecision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import multiprocessing.util
import collections
from typing import Dict, List, Tuple
from src.models import CRISPROfft, CnnCRISPR, CRISCross
from src.Datasets import MATCH_ROW_NUMBER1, MyDataModule, EPI_FEATURES, MAPPING, EPI_WEIGHTS
import json
from torchmetrics import Metric
from torchmetrics.functional import spearman_corrcoef
from torchmetrics.classification import MulticlassAveragePrecision
from pytorch_lightning.strategies import DDPStrategy


from torch.optim.lr_scheduler import LambdaLR
import hashlib


def short_hash(epi_features, length=6):
    # deterministic string representation
    s = ",".join(sorted(epi_features))
    h = hashlib.md5(s.encode()).hexdigest()
    return h[:10]

def get_base(config):
    epi_hash = short_hash(config["epi_features"], config["num_epi"])
    base_dir = f"RUNlogs/{config['experiment']}/test_split{config['split']}/ctl{config['context_layers']}_bs{config['batch_size']}_ws{config["windowsize"]}_ue{config['num_epi']}_seed{config['seed']}_hash{epi_hash}" 
    existing = os.listdir(os.path.join(base_dir, "run_")) if os.path.exists(base_dir) else []
    print(f"BASE DIR: {base_dir}")
    version = f"v{len(existing)}"
    return base_dir, version
    

def get_logger(config):
    base_dir, version = get_base(config)
    logger = TensorBoardLogger(
        save_dir=base_dir,   # your custom folder
        name=f"run_",
        version=f"v{version}"
    )   
    return logger


def estimate_stats(loader):

    for batch in loader:
        _, _, epi, _, _ = batch  # epi: (B, L, F)

        # move to CPU if needed
        epi = epi.detach().cpu()

        B, L, F = epi.shape
        epi = epi.view(-1, F)  # (B*L, F)

        if feature_sum is None:
            feature_sum = epi.sum(dim=0)
            feature_sq_sum = (epi ** 2).sum(dim=0)
        else:
            feature_sum += epi.sum(dim=0)
            feature_sq_sum += (epi ** 2).sum(dim=0)

        count += epi.shape[0]

    mean = feature_sum / count
    var = (feature_sq_sum / count) - mean**2
    std = torch.sqrt(var)
    return mean, var, std



class PreTrainModel(pl.LightningModule):
    def __init__(self, context_layers, hidden_dim, num_epi, dropout, seed, windowsize, merge,epi_weights, lr=1e-4, borders = None, ):
        super().__init__()
        
        if borders is not None:
            raise NotImplementedError("Not yet implemented")
            self.criterion = BarDistributionConfig(full_support=True, borders=borders).get_criterion()
            self.output_size = len(borders) - 1
            self.regression = True
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
            self.output_size = 1
            self.regression = False

        self.model = CRISCross(
            vocab_size=5,
            dropout=dropout,
            context_layers=context_layers,
            hidden_dim=hidden_dim,
            num_epi=num_epi,
            output_size=self.output_size,
            windowsize=windowsize,
            merge=merge

        )
        self.windowsize = windowsize
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.hparams.n_trainable_params = n_params
        self.hparams.seed = seed
        self.save_hyperparameters()


        self.lr = lr
        self.auprc = BinaryAveragePrecision()  # torchmetrics AUPRC
        self.test_auprc = BinaryAveragePrecision()
        self.alpha = nn.Parameter(torch.tensor(epi_weights), requires_grad=False)
        self.extra_epi_mask = False


        self.mask_prob = 0.2
        self.num_epi = num_epi

        self.max_idx = MAPPING.max()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.epi_loss_fn = nn.MSELoss(reduction="none")
        self.per_nt_classifier = nn.Linear(hidden_dim, 25)
        self.per_nt_epi_head = nn.Linear(hidden_dim, num_epi)
        self.auprc = MulticlassAveragePrecision( num_classes=25)
        self._first_batches_features = []



    
    def forward(self, target_x, off_target_x, epi, strands):
        _, logits = self.model(target_x, off_target_x, strands, epi, )
        epi_logits = self.per_nt_epi_head(logits)
        logits = self.per_nt_classifier(logits)
        return logits, epi_logits
    
    def mask_shit(self, batch):
        target_x, off_target_x, epi, y, counts, strands = batch
        batch_size, seq_len = target_x.shape
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        target_x = target_x.clone()


        # Generate a random mask
        mask_tensor = (torch.rand(batch_size, seq_len, device=target_x.device) < self.mask_prob)
        empty = mask_tensor.sum(dim=1) == 0
        if empty.any():
            idx = torch.randint(seq_len, (empty.sum(),), device=target_x.device)
            mask_tensor[empty, idx] = True
        # Expand mask to hidden dimension

        # Apply mask: set masked positions to zero
        target_x = target_x.masked_fill(mask_tensor, 0)
        off_target_x = off_target_x.clone()
        epi = epi.clone()
        off_target_x[:, center - 23//2 - 1:center+23//2] = off_target_x[:, center - 23//2 - 1:center+23//2].masked_fill(mask_tensor, 0)
        if len(epi.shape) > 1:
            epi[:, center - 23//2 - 1:center+23//2] = epi[:, center - 23//2 - 1:center+23//2].masked_fill(mask_tensor.unsqueeze(-1), 0)
            d = torch.randint(low=0, high=min(32, self.windowsize // 2), size=(1,))
            B, T = epi.shape[0:2]
            if self.extra_epi_mask:
                d = torch.randint(min(self.windowsize // 2, 128 // 2), min(128 // 2, self.windowsize // 2)+1, (B,), device=epi.device)
                idx = torch.arange(T, device=epi.device).unsqueeze(0) 
                epi_mask = (idx >= (center - d).unsqueeze(1)) & (idx < (center + d).unsqueeze(1))

                epi[epi_mask] = 0
        else:
            epi_mask = torch.zeros(epi.shape)
        return target_x, off_target_x, epi, mask_tensor
        

    def compute_tokenized_target(self, target_x, off_target_x, mask):
        bs, slen = target_x.shape
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2

        centered_off_target_x = off_target_x[:, center - 23//2 - 1:center+23//2].clone()


        y1 = torch.zeros((bs, slen), dtype=torch.long).to(target_x.device)
        y2 = torch.zeros((bs, slen), dtype=torch.long).to(target_x.device)
        y1[mask] = target_x[mask].to(torch.long)
        y2[mask] = centered_off_target_x[mask].to(torch.long)

        #y1[y1 > 1] -= 2
        #y2[y2 > 1] -= 2
        y = y1 * self.max_idx + y2
        return y




    def general_step(self, batch):
        target_x, off_target_x, epi, y, counts, strands = batch
        masked_target, masked_ot, epi_masked, mask = self.mask_shit(batch)
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        assert (off_target_x[:, center - 23//2 - 1:center+23//2] == target_x).sum(axis=1).min() >= 15
       
        bs, slen = target_x.shape
        y = self.compute_tokenized_target(target_x=target_x, off_target_x=off_target_x, mask=mask)
        if len(epi.shape) == 1:
            logits, epi_logits  = self(masked_target, masked_ot, None, strands)
            epi_loss = torch.zeros(epi_logits.shape, device=epi_logits.device)
        else:
            logits, epi_logits  = self(masked_target, masked_ot, epi_masked, strands)
            epi_loss = self.epi_loss_fn(epi_logits, epi[:, center - 23//2 - 1:center+23//2]) * mask[..., None]
        masked_loss = self.loss_fn(logits.flatten(start_dim=0, end_dim=1), y.flatten()) * mask[..., None].flatten()

        clsloss = masked_loss.sum() / mask.sum()

        epi_loss = epi_loss.sum(dim=(0,1)) / mask.sum()

        if self.training:
            for eidx in range(self.num_epi):
                self.log(f"train_epi_loss/epi{eidx}", epi_loss[eidx], on_step=False, on_epoch=True, prog_bar=True)
        epi_loss = (epi_loss * (self.alpha ** 2)).sum()

        loss = clsloss + epi_loss


        
        return loss, logits, epi_logits, y, mask, clsloss, epi_loss

    
    def training_step(self, batch, batch_idx):
        loss, logits, epi_logits, y, mask, clsloss, epiloss = self.general_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cls_loss", clsloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_epi_loss", epiloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("epi_logits_var", epi_logits[mask].std() ,on_step=False, on_epoch=True,)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, epi_logits, y, mask, clsloss, epiloss = self.general_step(batch)
        preds = torch.softmax(logits, dim=-1)
        # Update AUPRC metric
        self.auprc.update(preds[mask], y.int()[mask])
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_cls_loss", clsloss, prog_bar=True)
        self.log("val_epi_loss", epiloss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute AUPRC after whole epoch
        auprc_val = self.auprc.compute()
        self.log("val_auprc", auprc_val, prog_bar=True)
        self.auprc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999))

        # Total number of training steps
        train_loader = self.trainer.datamodule.train_dataloader()
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(5000, total_steps // 5)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                lr =  float(current_step) / float(max(1, warmup_steps))
            else:
                # After warmup, use cosine decay to zero
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

            return lr

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # update per step
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    


def run_pretraining(config):
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    embed_size = config["embed_size"]
    dropout= config["dropout"]
    epi_features = config["epi_features"]
    neighborhood_layers = config["context_layers"]
    seed = config["seed"]
    lr = config["lr"]
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features) if epi_features else 0
    epi_weights = 0
    config["num_epi"] = num_epi
    patience = config["patience"]
    train_test_split = config["split"]
    windowsize = config["windowsize"]
    merge = config["merge"]
    model_type = config["model_type"]

    pl.seed_everything(seed,workers=True)
    df = pd.read_csv("datasets/TCellDatasetWithextendedSequencesAndIDs.tsv", sep="\t")
    df = df[~pd.isna(df["AlphagenomeIndex"])]
    df = df[[col for col in df.columns if col in ["Strand", "Score", "Target_sequence", "Guide_sequence", "label", "AlphagenomeIndex", "GuideID", "extended_off_target", "ID"]]]
    run_settings = pd.read_csv("runSettings/RunSettingsLeaveOneOut.tsv", sep="\t")
    run_settings["val_set"] = run_settings["val_set"].apply(lambda y: None)
    run_settings["test_set"] = run_settings["test_set"].apply(lambda y: [])
    run_settings["exclude"] = run_settings["exclude"].apply(eval)
    dm = MyDataModule(df, 
        val_guides=run_settings.iloc[train_test_split]["val_set"], 
        test_guides=run_settings.iloc[train_test_split]["test_set"], 
        exclude=run_settings.iloc[train_test_split]["exclude"],
        num_samples=batch_size*100, batch_size=batch_size, 
        ag_dir="AGTensors3", 
        embedding_type="CRISPROfft" if model_type.lower() != "crosscrispr" else "raw", 
        windowsize=windowsize,
        epi_features=config["epi_features"],
        oversample=False,
        norm_epi=True if num_epi else False
    )


    
    model = PreTrainModel(
        context_layers=neighborhood_layers,
        hidden_dim=hidden_dim,
        num_epi=num_epi,
        dropout=dropout,
        lr=lr,
        seed=seed,
        windowsize=windowsize,
        merge=merge,
        epi_weights = epi_weights,
    )
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss", 
        mode="min", 
        save_top_k=1, 
        filename="best_model",
        save_last=True,
    )

    logger = get_logger(config=config)

    trainer = pl.Trainer(
        max_steps=50000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        callbacks=[checkpoint_cb],
        log_every_n_steps=10,
        logger=logger,
        deterministic=True,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=0.5,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),

    )
    trainer.fit(model, dm, ckpt_path=config["chkpt"] if "chkpt" in config else None)






if __name__ == "__main__":
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx is None:
        epi_features = [            
            "EX_ATAC",
            "EX_H3K4me1",  
            "EX_H3K4me3",  
            #"EX_H3K9ac", # not present in alphagenome
            "EX_H3K9me3",  
            "EX_H3K27ac",  
            "EX_H3K27me3",  
            "EX_H3K36me3",
            #"ATAC",
            #"DNASE", 
            #"RNA_SEQ",
            #"CHIP_HISTONE"
        ]
        epi_features = ['+_polyA plus RNA-seq',
            '+_total RNA-seq',
            '-_polyA plus RNA-seq',
            '-_total RNA-seq',
            'H3K27ac',
            'H3K27me3',
            'H3K36me3',
            'H3K4me1',
            'H3K9me3',
        ]
        params = {
            "batch_size": 512,
            "context_layers": 3,
            "hidden_dim": 512,
            "embed_size": 32,
            "dropout": 0.3,
            "epi_features": epi_features,
            "lr": 1e-4,
            "patience": 50,
            "seed": 42 * 0,
            "split": 0,
            "experiment": "PretrainingTest",
            "regression": False,
            "windowsize": 512,
            "merge": "early",
            "model_type": "crosscrispr",
            "accumulate_grad_batches": 2,
        }
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            required=True,
            help="Path to run_configs JSON file"
        )
        args = parser.parse_args()
        with open(args.config) as handle:
            config = json.load(handle)
        idx = int(idx)
        params = config[idx]
    run_pretraining(params)
    