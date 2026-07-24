
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
from torch.utils.data.sampler import WeightedRandomSampler
import torchmetrics
from sklearn.metrics import average_precision_score




MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                    "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16,"NA": 17, "NC": 18, "NG": 19, "NT": 20}

def get_input_encoding(string1,string2):

    # Iterate through pairs of characters from both strings
    numbers = [MATCH_ROW_NUMBER1[string1[i] + string2[i]] for i in range(len(string1))]

    return numbers



class cnnCRISPR(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, extension_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=40, bidirectional=True, batch_first=True)

        self.conv1 = nn.Conv1d(80, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(10)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(20, 40, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(40)
        self.conv4 = nn.Conv1d(40, 80, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(80)
        self.conv5 = nn.Conv1d(80, 100, kernel_size=5)
        self.bn5 = nn.BatchNorm1d(100)

        self.dropout1 = nn.Dropout(0.2)

        flattened_size = 100 * (3 + extension_size * 2)
        self.fc1 = nn.Linear(flattened_size, 20)

        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.dropout1(x)


        x = F.relu(self.fc1(x))

        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class LightningcnnCRISPR(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, extension_size,lr=0.01):
        super().__init__()
        self.model = cnnCRISPR(vocab_size, embed_size, extension_size)
        self.loss_fn = nn.CrossEntropyLoss()
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

        self.preds.append(y_hat.detach().cpu())  # Detach to avoid tracking gradients
        self.targets.append(y.detach().cpu())    # Detach targets as well
        
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
    train_inputs = train.apply(lambda row: get_input_encoding(row['sgRNA'], row['off_target']), axis = 1).to_list()
    train_labels = train["label"].values


    train_inputs = torch.tensor(train_inputs,dtype=torch.long)
    train_labels = torch.tensor(train_labels,dtype=torch.long)


    # val data
    val_inputs = val.apply(lambda row: get_input_encoding(row['sgRNA'], row['off_target']), axis = 1).to_list()
    val_labels = val["label"].values

    
    val_inputs = torch.tensor(val_inputs,dtype=torch.long)
    val_labels = torch.tensor(val_labels,dtype=torch.long)

    # test data
    test_inputs = test.apply(lambda row: get_input_encoding(row['sgRNA'], row['off_target']), axis = 1).to_list()
    test_labels = test["label"].values

    test_inputs = torch.tensor(test_inputs,dtype=torch.long)
    test_labels = torch.tensor(test_labels,dtype=torch.long)


    # create datasets
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    # samplers
    # Create weighted random sampler for bootstrap
    class_counts = torch.bincount(train_labels)  # Count occurrences of each class (0 and 1)
    class_weights = 1. / class_counts.float()  # Inverse frequency weighting
    sample_weights = class_weights[train_labels]  # Assign weights to each sample

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


    train_loader = DataLoader(train_dataset, batch_size=cfg["train_batch_size"], sampler=sampler,num_workers=7)
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



    
    model = LightningcnnCRISPR(
            vocab_size=21,
            embed_size=100,
            extension_size=cfg["extension_size"],
            lr=cfg["learning_rate"]) 
    

    torch.use_deterministic_algorithms(True, warn_only=True)

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True
    )
    #trainer.fit(model, train_loader, val_loader)


    ckpt_dir = 'checkpoints/' + cfg["output_dir"] + "/" + str(cfg["pytorch seed"]) + "/"
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    avg_ckpt_path = ckpt_files[0]

    model = LightningcnnCRISPR.load_from_checkpoint(
        checkpoint_path=avg_ckpt_path,
        vocab_size=21,
        embed_size=100,
        extension_size=cfg["extension_size"],
        lr=cfg["learning_rate"]
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
    "train_batch_size": 256,
    "val_batch_size": 256,
    "test_batch_size": 256,
    "train/test_split_seed": 42,
    "learning_rate": 0.01,
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
print(df)

# save_path = os.path.join(parent_dir, "results.csv")
save_path = "cnnCRISPR_cross_cell_results_fixed_k562.csv"

df.to_csv(save_path, index=False)

print(f"Results saved to {save_path}")