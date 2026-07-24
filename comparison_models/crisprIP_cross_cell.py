import os
import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import torchmetrics
from sklearn.metrics import average_precision_score




encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0], '-': [0, 0, 0, 0]}
pos_dict = {'A':1, 'T':2, 'G':3, 'C':4, '_':5, '-':5}


def my_encode_on_off_dim(target_seq, off_target_seq,length=0):

    tlen = length*2 + 24
    target_seq = "-" *(tlen-len(target_seq)) + target_seq
    
    off_target_seq = "-" *(tlen-len(off_target_seq)) + off_target_seq
    target_seq_code = np.array([encoded_dict[base] for base in list(target_seq)])
    off_target_seq_code = np.array([encoded_dict[base] for base in list(off_target_seq)])
    on_off_dim6_codes = []
    for i in range(len(target_seq)):
        diff_code = np.bitwise_or(target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code*-1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    #isPAM = np.zeros((24,1))
    isPAM = np.zeros((tlen,1))
    # isPAM = np.zeros((length,1))
    #isPAM[-3:, :] = 1
    #isPAM = np.zeros((tlen,1))
    #isPAM[21+length:24+length, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code


def get_class_ratio(df):
    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    ratio = neg_count / pos_count if pos_count > 0 else float('inf')
    return pos_count, neg_count, ratio




class CrisprIPModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # Conv layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=60,
            kernel_size=(1, 7)
        )

        # BiLSTM (60 avg + 60 max)
        self.lstm = nn.LSTM(
            input_size=120,
            hidden_size=30,
            bidirectional=True,
            batch_first=True
        )

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=60,
            num_heads=1,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(120, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 2)

        self.dropout = nn.Dropout(0.9)

    def forward(self, x):

        # (B, L) → (B,1,L,1)
        x = x.unsqueeze(1)

        # Conv2D
        x = self.conv1(x)            # (B,60,L,1)
        x = x.squeeze(3)             # (B,60,L)

        # transpose for pooling/LSTM
        x = x.transpose(1,2)         # (B,L,60)

        # pool over sequence dimension
        x_pool = x.transpose(1,2)    # (B,60,L)

        avg_pool = F.avg_pool1d(x_pool, 2, 2)   # (B,60,L/2)
        max_pool = F.max_pool1d(x_pool, 2, 2)   # (B,60,L/2)

        # concat features
        x = torch.cat([avg_pool, max_pool], dim=1)   # (B,120,L/2)

        # prepare for LSTM
        x = x.transpose(1,2)          # (B,L/2,120)

        # BiLSTM
        x, _ = self.lstm(x)           # (B,L/2,60)

        # Attention
        attn_out, _ = self.attention(x, x, x)   # (B,L/2,60)

        # Global pooling
        avg = F.adaptive_avg_pool1d(attn_out.transpose(1,2), 1).squeeze(-1)
        mx  = F.adaptive_max_pool1d(attn_out.transpose(1,2), 1).squeeze(-1)

        x = torch.cat([avg, mx], dim=1)   # (B,120)

        # FC layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc3(x)

        return x


class LightningcrisprIP(pl.LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.model = CrisprIPModel()
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")

        self.preds_epoch = []
        self.targets_epoch = []

        self.preds = []
        self.targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

        y_probs = torch.softmax(y_hat, dim=-1)[:, 1]
        self.preds_epoch.append(y_probs.detach().cpu())
        self.targets_epoch.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.preds_epoch, dim=0)
        targets = torch.cat(self.targets_epoch, dim=0)

        val_auprc = self.val_auprc(preds, targets.int())
        self.log('epoch_val_auprc', val_auprc, prog_bar=True)

        self.preds_epoch.clear()
        self.targets_epoch.clear()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.preds.append(y_hat.detach().cpu())
        self.targets.append(y.detach().cpu())

        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)




def train_model_cross_cell(cfg):

    pl.seed_everything(cfg["pytorch seed"],workers=True)

    df = pd.read_csv("../datasets/Tcell_AG+histones+EX_compare_2bit.csv")
    df = df.rename(columns={'target': 'sgRNA'})
    df["label"] = df["label"].astype(int)
    k562 = pd.read_csv("../datasets/k562_deepcrispr_withCoords_hg38.csv")


    trainval = df
    test = k562
 
    train, val = train_test_split(trainval, test_size=0.2, random_state=cfg["train/test_split_seed"], stratify=trainval["label"])
    
    # train data
    train_inputs = np.array(train.apply(lambda row: my_encode_on_off_dim(row['sgRNA'], row['off_target'],cfg["extension_size"]), axis = 1).to_list())
    train_labels = train["label"].values


    ################################################################


    train_inputs = torch.tensor(train_inputs,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.long)


    # val data
    val_inputs = np.array(val.apply(lambda row: my_encode_on_off_dim(row['sgRNA'], row['off_target'],cfg["extension_size"]), axis = 1).to_list())
    val_labels = val["label"].values

    
    val_inputs = torch.tensor(val_inputs,dtype=torch.float32)
    val_labels = torch.tensor(val_labels,dtype=torch.long)

    # test data
    test_inputs = np.array(test.apply(lambda row: my_encode_on_off_dim(row['sgRNA'], row['off_target'],cfg["extension_size"]), axis = 1).to_list())
    test_labels = test["label"].values

    test_inputs = torch.tensor(test_inputs,dtype=torch.float32)
    test_labels = torch.tensor(test_labels,dtype=torch.long)


    # create datasets
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)


    train_loader = DataLoader(train_dataset, batch_size=cfg["train_batch_size"], shuffle=True,num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=cfg["val_batch_size"], num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=cfg["test_batch_size"],num_workers=7)


    # EarlyStopping callback 
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='epoch_val_auprc',
        patience=cfg["patience"],
        mode='max')
    # checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='epoch_val_auprc',  #
        dirpath='checkpoints/'+ cfg["output_dir"]+ "/" + str(cfg["pytorch seed"]) + "/",  # Directory to save checkpoints
        filename='{epoch}-{epoch_val_auprc:.2f}',  # S
        save_top_k=1,  # Save only the best model
        mode='max',
        save_weights_only=True) # S


    
    model = LightningcrisprIP(
            lr=cfg["learning_rate" ]) # Set to True if you want to use side features)
    

    torch.use_deterministic_algorithms(True, warn_only=True)


    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True
    )
    trainer.fit(model, train_loader, val_loader)
    ckpt_dir = 'checkpoints/' + cfg["output_dir"] + "/" + str(cfg["pytorch seed"]) + "/"
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    avg_ckpt_path = ckpt_files[0] 



    model = LightningcrisprIP.load_from_checkpoint(
        checkpoint_path=avg_ckpt_path,
        lr=cfg["learning_rate"],
    )

    trainer.predict(model,test_loader)

    # collect preds 
    all_preds = torch.cat(model.preds, dim=0)
    all_targets = torch.cat(model.targets, dim=0)


    softmax = torch.nn.Softmax(dim=1)  
    all_probs = softmax(all_preds)
    positive_class_probs = all_probs[:, 1]

    aucpr =  average_precision_score(all_targets,positive_class_probs)
    print(aucpr)


    
    return aucpr


    


base_cfg = {
    "output_dir": "cross_cell",
    "extension_size": 0,
    "train_batch_size": 4000,
    "val_batch_size": 4000,
    "test_batch_size": 4000,
    "train/test_split_seed": 42,
    "learning_rate": 0.001,
    "patience": 15,
    "epochs": 100,
    "pytorch seed": 42,
}



seeds = list(range(25))

parent_dir = "cross_cell_results_fixed_k562"
os.makedirs(parent_dir, exist_ok=True)

base_cfg["output_dir"] = parent_dir

# run over seeds and save results to csv

results = []

for seed in seeds:
    base_cfg["pytorch seed"] = seed
    base_cfg["train/test_split_seed"] = seed

    
    aucpr = train_model_cross_cell(base_cfg)
    
    results.append({
        "seed": seed,
        "aucpr": aucpr
    })

df = pd.DataFrame(results)

# save_path = os.path.join(parent_dir, "results.csv")
save_path = "crisprIP_cross_cell_results_fixed_k562.csv"
df.to_csv(save_path, index=False)

print(f"Results saved to {save_path}")