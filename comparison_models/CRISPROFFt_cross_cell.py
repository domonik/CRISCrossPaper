import os
import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
import torchmetrics
from sklearn.metrics import average_precision_score



pl.seed_everything(42,workers=True)




MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                    "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16,"NA": 17, "NC": 18, "NG": 19, "NT": 20}

def get_input_encoding(string1,string2):
    # Iterate through pairs of characters from both strings
    numbers = [MATCH_ROW_NUMBER1[string1[i] + string2[i]] for i in range(len(string1))]
    return numbers



class CustomAttention(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
        self.x_transform = nn.Linear(self.time_steps, self.time_steps)
        self.g_transform = nn.Linear(self.time_steps, self.time_steps)
        nn.init.uniform_(self.x_transform.weight, -0.1, 0.1)
        nn.init.uniform_(self.g_transform.weight, -0.1, 0.1)

        self.a_transform = nn.Linear(self.time_steps, self.time_steps, bias=False)


    def forward(self, x, g):
        # x and g: (batch_size, time_steps, input_dim)
        # input_dim = x.size(2)
    
        # Permute and reshape
        # print(x.shape)
        # print(g.shape)
        x1 = x.permute(0, 2, 1)  # (batch_size, input_dim, time_steps)
        g1 = g.permute(0, 2, 1)  # (batch_size, input_dim, time_steps)
        # 
        # ##################seems like no reshaping is needed, due to difference between keras and pytorch
        # print(x1.shape)
        # print(g1.shape)

        x2 = self.x_transform(x)
     
        g2 = self.g_transform(g)
 
        # Add transformed tensors
        x3 = x2 + g2
        
        # Generate attention weights
        a = self.a_transform(x3)
        a_probs = F.softmax(a, dim=-1)


        # Reshape and multiply
        #a_probs = a_probs.permute(0, 2, 1)
        output_attention_mul = x * a_probs
    
        return output_attention_mul


class CRISPROfft(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, extension_size):
        super().__init__()


        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.custom_attention = CustomAttention(time_steps=11 + (extension_size * 2))

        # Convolutional Layers
        self.conv1 = nn.Conv1d(embed_size, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(20)

        self.conv2 = nn.Conv1d(20, 40, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(40)

        self.conv3 = nn.Conv1d(40, 80, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(80)

        # Extra conv for attention
        self.conv11 = nn.Conv1d(20, 80, kernel_size=9)

        # Fully-connected sizes
        flattened_size = 80 * (11 + (extension_size * 2))

        self.fc1 = nn.Linear(flattened_size, 40)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(40, 20)
        self.dropout2 = nn.Dropout(0.2)

        # Final output
        self.output = nn.Linear(20, 2)

    def forward(self, x):

        # Embedding
        x = self.embedding(x).transpose(1, 2)

        # Convolutions
        conv1_out = self.bn1(F.relu(self.conv1(x)))
        conv2_out = self.bn2(F.relu(self.conv2(conv1_out)))
        conv3_out = self.bn3(F.relu(self.conv3(conv2_out)))

        # Attention conv
        conv11_out = self.conv11(self.bn1(F.relu(self.conv1(x))))

        # Attention
        attended = self.custom_attention(conv11_out, conv3_out)

        # Flatten
        x = torch.flatten(attended, 1)

        # First FC
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        # Second FC
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Output logits
        return self.output(x)


class LightningcnnCRISPR(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, extension_size,lr=0.01):
        super().__init__()
        self.model = CRISPROfft(vocab_size, embed_size, extension_size)
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
        #y_hat = self(x, x_epi)
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


    #train_loader = DataLoader(train_dataset, batch_size=cfg["train_batch_size"], shuffle=True,num_workers=7)
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
        save_top_k=1,  
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
    trainer.fit(model, train_loader, val_loader)

    ckpt_dir = 'checkpoints/' + cfg["output_dir"] + "/" + str(cfg["pytorch seed"]) + "/"
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    avg_ckpt_path = ckpt_files[0]


    model = LightningcnnCRISPR.load_from_checkpoint(
        checkpoint_path=avg_ckpt_path,
        vocab_size=21,
        embed_size=100,
        extension_size=cfg["extension_size"],
        lr=cfg["learning_rate"],
    )

    trainer.predict(model,test_loader)

    # collect preds 
    all_preds = torch.cat(model.preds, dim=0)
    all_targets = torch.cat(model.targets, dim=0)

    # Assuming a binary classification, convert logits to probabilities using softmax
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
    "train/test_split_seed": 10,
    "learning_rate": 1e-3,
    "patience": 15,
    "epochs": 100,
    "pytorch seed": 42, 
}



seeds = list(range(25))



parent_dir = "cross_cell_results_fixed_k562"
os.makedirs(parent_dir, exist_ok=True)

base_cfg["output_dir"] = parent_dir



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

#save_path = os.path.join(parent_dir, "results.csv")
save_path = "CRISPROFFt_cross_cell_results_fixed_k562.csv"
df.to_csv(save_path, index=False)

print(f"Results saved to {save_path}")

