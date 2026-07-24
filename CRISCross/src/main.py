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
from src.models import CRISPROfft, CnnCRISPR, CRISCross, CrisprIP
from src.Datasets import MATCH_ROW_NUMBER1, MyDataModule, EPI_FEATURES
import json
from torchmetrics import Metric
from torchmetrics.functional import spearman_corrcoef
from src.pretrain import PreTrainModel, get_logger

from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryRecallAtFixedPrecision


class SpearmanCorr(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return spearman_corrcoef(preds, targets)




class PLCRISPRWrapper(pl.LightningModule):
    def __init__(self, model_type, embed_size, context_layers, hidden_dim, num_epi, dropout, seed, windowsize, merge, lr=1e-4, borders = None):
        super().__init__()
        
        if borders is not None:
            raise NotImplementedError()
            self.criterion = BarDistributionConfig(full_support=True, borders=borders).get_criterion()
            self.output_size = len(borders) - 1
            self.regression = True
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
            self.output_size = 1
            self.regression = False



        if model_type.lower() == "crisprofft":
            self.model = CRISPROfft(
                vocab_size=len(MATCH_ROW_NUMBER1), 
                dropout=dropout, 
                context_layers=context_layers,
                embed_size=embed_size,
                hidden_dim=hidden_dim, 
                num_epi=num_epi,
                output_size=self.output_size,
                windowsize=windowsize,
                merge=merge
                )
        elif model_type.lower() == "cnncrispr":
            self.model = CnnCRISPR(len(MATCH_ROW_NUMBER1), 
                embed_size, 
                dropout=dropout, 
                context_layers=context_layers, 
                hidden_dim=hidden_dim, 
                num_epi=num_epi,
                output_size=self.output_size,
                windowsize=windowsize,
                merge=merge
                )
        elif model_type.lower() == "crisprip":
            self.model = CrisprIP(
                dropout=dropout, 
                num_epi=num_epi,
                output_size=self.output_size,
                windowsize=windowsize,
                merge=merge
                )
        elif model_type.lower() == "crosscrispr":
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

        else:
            raise ValueError("Wrong model type")
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.hparams.n_trainable_params = n_params
        self.hparams.seed = seed
        self.save_hyperparameters()

        self.lr = lr
        self.auprc = BinaryAveragePrecision()  # torchmetrics AUPRC
        self.test_auprc = BinaryAveragePrecision()
        self.train_spearman = SpearmanCorr()
        self.val_spearman = SpearmanCorr()
        self.test_spearman = SpearmanCorr()
        self.recall90 = BinaryRecallAtFixedPrecision(
            min_precision=0.90
        )
    
    def forward(self, target_x, off_target_x, epi, strands):
        return self.model(target_x, off_target_x, strands, epi)

    def general_step(self, batch):
            
        target_x, off_target_x, epi, y, counts, strands = batch
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        t_val = (off_target_x[:, center - 23//2 - 1:center+23//2] == target_x).sum(axis=1).min()
        assert t_val >= 15, f"Min of {t_val} detected"


        res = self(target_x, off_target_x, epi, strands)
        logits = res[0] if isinstance(res, tuple) else res
        logits = logits.squeeze(1)


        if self.regression:
            loss = self.criterion(logits[None, ...], counts.float()).mean()
            preds = self.criterion.mean(logits)
        else:
           
            loss = self.criterion(logits, y.float()).mean()
            preds = torch.sigmoid(logits)
        return loss, preds




    
    def training_step(self, batch, batch_idx):
        loss, preds = self.general_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        y, counts, strands = batch[3:]
        loss, preds = self.general_step(batch)
        
        # Update AUPRC metric
        self.auprc.update(preds, y.int())
        if self.regression:
            preds = preds[y == 1]
            counts = counts[y == 1]
            self.val_spearman.update(preds, counts)

        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_test_start(self):
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        y, counts, strands = batch[3:]
        loss, preds = self.general_step(batch)
        
        # Update AUPRC metric
        self.test_auprc.update(preds, y.int())
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.recall90.update(preds, y.int())
        if self.regression:
            preds = preds[y == 1]
            counts = counts[y == 1]

            self.test_spearman.update(preds, counts)
        # Update test AUPRC metric
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        if self.regression:
            spearman = self.test_spearman.compute()
            self.log("test_spearman", spearman, prog_bar=True)
            self.test_spearman.reset()
        auprc_val = self.test_auprc.compute()
        self.log("test_auprc", auprc_val, prog_bar=True)
        recall90 = self.recall90.compute()
        recall, precision = recall90
        self.log("test_recall@90precision", recall)
        self.log("test_precision@90precision", precision)
        self.recall90.reset()
        self.auprc.reset()
        
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        ks = [10, 50, 100, 500, 1000]
        for k in ks:
            if len(preds) < k:
                precision_at_k = torch.tensor(float("nan"))
            idx = torch.argsort(preds, descending=True)[:k]

            precision_at_k = targets[idx].float().mean()
            self.log(f"test_precision@{k}", precision_at_k)
        
    
    def on_validation_epoch_end(self):
        if self.regression:
            spearman = self.val_spearman.compute()
            self.log("val_spearman", spearman, prog_bar=True)
            # Reset for safety
            self.val_spearman.reset()
        auprc_val = self.auprc.compute()
        self.log("val_auprc", auprc_val, prog_bar=True)
        self.auprc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999))

        # Total number of training steps
        train_loader = self.trainer.datamodule.train_dataloader()
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(10, total_steps // 5)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # update per step
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def get_distribution_borders(datamodule):
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    ys = []
    for idx, batch in enumerate(train_loader):
        target_x, off_target_x, epi, y, counts, strands = batch      
        if idx == 100:
            break
        ys.append(counts)
    ys = torch.concat(ys)
    borders = [0.] + bar_distribution.get_bucket_borders(num_outputs=15, ys=ys).tolist() + [1.0]
    borders = torch.tensor(borders).unique().tolist()
    print(f"Using these borders {borders}")
    return borders




def run_training(config):
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    embed_size = config["embed_size"]
    dropout= config["dropout"]
    epi_features = config["epi_features"]
    context_layers = config["context_layers"]
    seeds = config["seed"]
    lr = config["lr"]
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)
    config["num_epi"] = num_epi
    patience = config["patience"]
    train_test_split = config["split"]
    windowsize = config["windowsize"]
    merge = config["merge"]
    model_type = config["model_type"]
    run_settings = pd.read_csv(config["run_settings"], sep="\t")
    run_settings["val_set"] = run_settings["val_set"].apply(eval)
    run_settings["test_set"] = run_settings["test_set"].apply(eval)
    run_settings["exclude"] = run_settings["exclude"].apply(eval)



    if isinstance(seeds, int):
        seeds = [seeds]
    else:
        seeds = config["seed"]
    for seed in seeds:
        config["seed"] = seed
        df = pd.read_csv(config["dataset"], sep="\t")



        pl.seed_everything(seed,workers=True)
        if "AlphagenomeIndex" in df:
            df = df[~pd.isna(df["AlphagenomeIndex"])]
        else:
            df["AlphagenomeIndex"] = 1
            assert num_epi == 0, "No epi features are allowed if AlphagenomeIndex is not set"
        if num_epi == 0:
            assert merge is None


        if model_type.lower() == "crisprofft":
            embedding_type = "CRISPROfft"
        elif model_type.lower() == "crosscrispr":
            embedding_type = "raw"
        elif model_type.lower() == "cnncrispr":
            embedding_type = "CRISPROfft"
        elif model_type.lower() == "crisprip":
            embedding_type = "crisprip"


        dm = MyDataModule(df, 
            val_guides=run_settings.iloc[train_test_split]["val_set"], 
            test_guides=run_settings.iloc[train_test_split]["test_set"], 
            exclude=run_settings.iloc[train_test_split]["exclude"],
            num_samples=batch_size*10, batch_size=batch_size, 
            ag_dir=config.get("epi_dir", "AGTensors3"), 
            embedding_type=embedding_type, 
            windowsize=windowsize,
            epi_features=config["epi_features"]
        )

        if config["regression"]:
            borders = get_distribution_borders(dm)
        else:
            borders = None

        model = PLCRISPRWrapper(
            model_type=model_type,
            embed_size=embed_size,
            context_layers=context_layers,
            hidden_dim=hidden_dim,
            num_epi=num_epi,
            dropout=dropout,
            lr=lr,
            seed=seed,
            borders=borders,
            windowsize=windowsize,
            merge=merge

        )
        if "chkpt" in config:
            ptm = PreTrainModel.load_from_checkpoint(config["chkpt"])
            print("Using pretrained model")
            model.model.load_state_dict(ptm.model.state_dict())

        checkpoint_cb = ModelCheckpoint(
            monitor="val_auprc", 
            mode="max", 
            save_top_k=1, 
            filename="best_model"
        )
        earlystop_cb = EarlyStopping(monitor="val_auprc", mode="max", patience=patience)

        logger = get_logger(config=config)

        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_cb, earlystop_cb],
            log_every_n_steps=10,
            logger=logger,
            deterministic="warn",
            gradient_clip_val=0.5,
            accumulate_grad_batches=config["accumulate_grad_batches"] if "accumulate_grad_batches" in config else 1
        )


        trainer.fit(model, dm)
        best_model = PLCRISPRWrapper.load_from_checkpoint(checkpoint_cb.best_model_path)
        best_aucpr = trainer.callback_metrics["val_auprc"].item()
        test_result = trainer.test(best_model, dm)[0]

if __name__ == "__main__":
    import argparse
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx is None:
        epi_features = [            
            #"EX_ATAC",
            #"EX_H3K4me1",  
            #"EX_H3K4me3",  
            #"EX_H3K9ac", # not present in alphagenome
            #"EX_H3K9me3",  
            #"EX_H3K27ac",  
            #"EX_H3K27me3",  
            #"EX_H3K36me3",
            "ATAC",
            #"DNASE", 
            #"RNA_SEQ",
            "CHIP_HISTONE"
        ]
        #epi_features = []
        epi_features = [            
            "EX_ATAC",
            "EX_H3K4me1",  
            "EX_H3K4me3",  
            "EX_H3K9ac", # not present in alphagenome
            "EX_H3K9me3",  
            "EX_H3K27ac",  
            "EX_H3K27me3",  
            "EX_H3K36me3",
            "ATAC",
            "DNASE", 
            "RNA_SEQ",
            "CHIP_HISTONE"
        ]
        epi_features = [            
            "EX_ATAC",
            "EX_H3K4me1",  
            "EX_H3K4me3",  
            "EX_H3K9ac", # not present in alphagenome
            "EX_H3K9me3",  
            "EX_H3K27ac",  
            "EX_H3K27me3",  
            "EX_H3K36me3",
        ]
        epi_features = [
            "ATAC",
            "H3K4me1",
            "H3K4me3",
            "H3K9me3",
            "H3K27ac",
            "H3K27me3",
            "H3K36me3"
            ]
        params = {
            "batch_size": 128,
            "context_layers": 3,
            "hidden_dim": 512,
            "embed_size": 62,
            "dropout": 0.2,
            "epi_features": epi_features,
            "lr": 1e-4,
            "patience": 10,
            "seed": 42 * 0,
            "split": 1,
            "experiment": "TestRUN",
            "regression": False,
            "windowsize": 23,
            "merge": None, #"early",
            "model_type": "crosscrispr",
            "chkpt": "RUNlogs/PretrainingPaperFixed/AG/test_split0/ctl3_bs512_ws23_ue7_seed0_hashe5da1f09f1/run_/vv0/checkpoints/last.ckpt",
            "accumulate_grad_batches": 2,
            #"dataset": "Hek293WithextendedSequences.tsv",
            "run_settings": "runSettings/RunSettingsLeaveOneOut.tsv",
            "dataset": "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv",
        }
    else:
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
    run_training(params)



