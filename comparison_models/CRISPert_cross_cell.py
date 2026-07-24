import os
import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchmetrics
import collections
from typing import Optional
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import average_precision_score
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from transformers import BertPreTrainedModel
from transformers import(
AdamW,
BertConfig
)
from transformers.models.bert.modeling_bert import BertModel
from modified_transformer_models import BertSmallForSequenceClassificationFeatures_late_crosscell


class SimpleDNATokenizer:
    def __init__(self, vocab: Dict[str, int], cls_token="[CLS]", unk_token="[UNK]",sep_token="[SEP]"):
        """
        Initializes the SimpleDNATokenizer.

        Args:
            vocab (Dict[str, int]): A dictionary mapping k-mers and special tokens to their corresponding IDs.
            cls_token (str): The classification token to be added at the beginning of the sequence.
            unk_token (str): The unknown token for k-mers not found in the vocabulary.
        """
        self.vocab = vocab
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token_id = self.vocab[cls_token]  # Assumes [CLS] is in the vocab
        self.unk_token_id = self.vocab[unk_token]  # Assumes [UNK] is in the vocab
        self.sep_token_id = self.vocab[sep_token]  # Assumes [SEP] is in the vocab

    def tokenize(self, kmer_sequence: List[str]) -> Tuple[List[int], List[int]]:
        """
        Tokenizes a sequence of k-mers and adds a [CLS] token at the beginning.

        Args:
            kmer_sequence (List[str]): A list of k-mers to be tokenized.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing the input IDs and attention mask.
        """
        # Convert k-mers to their corresponding IDs
        input_ids = [self.vocab.get(kmer, self.unk_token_id) for kmer in kmer_sequence]
        
        # Add [CLS] token at the beginning
        input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        
        # Create attention mask (1 for all tokens, since no padding is needed)
        attention_mask = [1] * len(input_ids)
        
        return input_ids, attention_mask

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers.

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- list, list of kmers
    """
    kmers = [seq[x:x+k] for x in range(len(seq) + 1 - k)]

    return kmers


def encode_seq(target,off_target):

    encode_dict ={
        ("A","A"):"A",
        ("A","C"):"Z",
        ("A","G"):"Y",
        ("A","T"):"X",
        ("C","C"):"C",
        ("C","A"):"W",
        ("C","G"):"V",
        ("C","T"):"U",
        ("G","G"):"G",
        ("G","A"):"S",
        ("G","C"):"R",
        ("G","T"):"L",
        ("T","T"):"T",
        ("T","A"):"Q",
        ("T","C"):"P",
        ("T","G"):"O",
    }
    
    new_seq = ""
    for char in zip(target,off_target):
        new_seq += encode_dict[char]
   
    return(new_seq)

class BertSmallForSequenceClassification_late_crosscell(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel.from_pretrained(config._name_or_path, config=config)
   
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


        self.post_init()

    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
    
        logits = self.classifier(pooled_output)
 

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class CRISPertFeatures(pl.LightningModule):
    def __init__(self,model_path,config):
        super().__init__()
        self.bert_model = BertSmallForSequenceClassificationFeatures_late_crosscell.from_pretrained(model_path,config=config,cache_dir=None)
       
    def forward(self, input_ids, attention_mask):

        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        
        return outputs
    


class LightningCRISPertFeatures(pl.LightningModule):
    def __init__(self, cfg,bert_config):
        super().__init__()
        self.config = bert_config
        self.config.hidden_dropout_prob = cfg["hidden_dropout_prob"]
        self.config.attention_probs_dropout_prob = cfg["attention_probs_dropout_prob"]
        self.model = CRISPertFeatures(cfg["model_name_or_path"],config=self.config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = cfg["learning_rate"]
        self.adam_epsilon = cfg["adam_epsilon"] 
        self.beta1 = cfg["beta1"]
        self.beta2 = cfg["beta2"]
        

        self.train_acc = torchmetrics.Accuracy(task="multiclass",num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",num_classes=2)
        self.val_auprc = torchmetrics.AveragePrecision(task="binary") 
        self.train_auprc = torchmetrics.AveragePrecision(task="binary")

        
        # Store predictions and targets
        # for prediction
        self.preds = []
        self.targets = []

        # for validation
        self.preds_epoch = []
        self.targets_epoch = []



    def forward(self, x_input_ids,x_attention):
        return self.model(x_input_ids,x_attention)



    def training_step(self, batch, batch_idx):


        x_input_ids,x_attention, y = batch
        y_hat = self(x_input_ids,x_attention)[0]

        loss = self.loss_fn(y_hat,y)

        self.log('train_loss', loss, prog_bar=True, logger=True)

        # Accuracy
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, prog_bar=True, logger=True)

        # AUCPR
        y_probs = torch.softmax(y_hat, dim=-1)[:, 1]  # probability of positive class
        self.train_auprc.update(y_probs, y.int())
        self.log('train_auprc', self.train_auprc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss
    

    def validation_step(self, batch, batch_idx):

        x_input_ids,x_attention, y = batch
        y_hat = self(x_input_ids,x_attention)[0]

        loss = self.loss_fn(y_hat,y)

        self.log('val_loss', loss, prog_bar=True, logger=True)  # Log validation loss
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, prog_bar=True, logger=True)  # Log validation accuracy
        
        y_probs = torch.softmax(y_hat, dim=-1)[:, 1]  # Get probabilities for the positive class
 
        self.preds_epoch.append(y_probs.detach().cpu())  # Detach to avoid tracking gradients
        self.targets_epoch.append(y.detach().cpu())      # Detach targets as well
        return loss
    
    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        preds = torch.cat(self.preds_epoch, dim=0)
        targets = torch.cat(self.targets_epoch, dim=0)

        # Compute AUCPR for the entire epoch
        val_auprc = self.val_auprc(preds, targets.int())
        self.log('epoch_val_auprc', val_auprc, prog_bar=True, logger=True)

        # Clear stored predictions and targets for the next epoch
        self.preds_epoch.clear()
        self.targets_epoch.clear()
       

    def predict_step(self, batch, batch_idx):
        # Assuming the batch is a tuple (x, y), where x is the input tensor
        x_input_ids,x_attention, y = batch
    
        y_hat = self(x_input_ids,x_attention)[0]
        
        # Store predictions and targets
        self.preds.append(y_hat.detach().cpu())  # Detach to avoid tracking gradients
        self.targets.append(y.detach().cpu())    # Detach targets as well
        
        return y_hat

    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(self.parameters(), lr=self.lr,eps=self.adam_epsilon,betas=(self.beta1, self.beta2),weight_decay=0)
        
        return [optimizer]
    


def encode_and_tokenize_seq(file_path):

    df = pd.read_csv(file_path)

    df = df.rename({"target": "sgRNA",}, axis=1)

    df["label"] = df['label'].astype(int)
    df["comb"]  = df.apply(lambda x: encode_seq(x["sgRNA"],x["off_target"]),axis=1)
    df["comb"] = df.apply(lambda x: seq2kmer(x["comb"],3),axis=1)

    # Tokenise sequence
    vocab = load_vocab("vocab.txt")
    tokenizer = SimpleDNATokenizer(vocab)

    # Apply tokenizer to each row in "comb" column
    df["tokenized"] = df["comb"].apply(lambda x: tokenizer.tokenize(x))

    # Extract input_ids and attention_mask into separate columns
    df["input_ids"] = df["tokenized"].apply(lambda x: x[0])  # First element of tuple
    df["attention_mask"] = df["tokenized"].apply(lambda x: x[1])  # Second element of tuple

    # Drop the intermediate "tokenized" column if not needed
    df.drop(columns=["tokenized"], inplace=True)
    df.drop(columns=["comb"], inplace=True)
    print(df.columns)

    return df


def encode_and_tokenize_seq_u20s(file_path):


    df= pd.read_csv(file_path,sep="\t")
    df = df.rename(columns={'target': 'sgRNA', "offtarget_sequence": "off_target"})

    df["label"] = df['label'].astype(int)
    df["comb"]  = df.apply(lambda x: encode_seq(x["sgRNA"],x["off_target"]),axis=1)
    df["comb"] = df.apply(lambda x: seq2kmer(x["comb"],3),axis=1)

    # Tokenise sequence
    vocab = load_vocab("vocab.txt")
    tokenizer = SimpleDNATokenizer(vocab)

    # Apply tokenizer to each row in "comb" column
    df["tokenized"] = df["comb"].apply(lambda x: tokenizer.tokenize(x))

    # Extract input_ids and attention_mask into separate columns
    df["input_ids"] = df["tokenized"].apply(lambda x: x[0])  # First element of tuple
    df["attention_mask"] = df["tokenized"].apply(lambda x: x[1])  # Second element of tuple

    # Drop the intermediate "tokenized" column if not needed
    df.drop(columns=["tokenized"], inplace=True)
    df.drop(columns=["comb"], inplace=True)
    print(df.columns)

    return df


def to_tensors(inputs,mask,labels):
    # Convert 'input_ids' and 'attention_mask' to tensors
    inputs = torch.tensor(inputs, dtype=torch.long)
    masks = torch.tensor(mask, dtype=torch.long)

    labels = torch.tensor(labels, dtype=torch.long)
    
    return inputs, masks,labels





def train_model_cross_cell(cfg):

    pl.seed_everything(cfg["pytorch seed"],workers=True)


    vocab = load_vocab("vocab.txt")

    file_path = "../datasets/Tcell_AG+histones+EX_compare_2bit.csv"
    file_path2 = "../datasets/k562_deepcrispr_withCoords_hg38.csv"

    tokenizer = SimpleDNATokenizer(vocab)

    df = encode_and_tokenize_seq(file_path)
    k562 = encode_and_tokenize_seq(file_path2)


    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=23,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )




    trainVal = df
    test = k562

    train, val = train_test_split(trainVal, test_size=0.2, random_state=cfg["train/test_split_seed"], stratify=trainVal["label"])


    print(train.columns)
    
    train_inputs = torch.tensor(train["input_ids"].to_list(), dtype=torch.long)
    train_masks = torch.tensor(train["attention_mask"].to_list(), dtype=torch.long)
    train_labels = torch.tensor(train["label"].to_list(), dtype=torch.long)

    val_inputs = torch.tensor(val["input_ids"].to_list(), dtype=torch.long)
    val_masks = torch.tensor(val["attention_mask"].to_list(), dtype=torch.long)
    val_labels = torch.tensor(val["label"].to_list(), dtype=torch.long)

    test_inputs = torch.tensor(test["input_ids"].to_list(), dtype=torch.long)
    test_masks = torch.tensor(test["attention_mask"].to_list(), dtype=torch.long)
    test_labels = torch.tensor(test["label"].to_list(), dtype=torch.long)
    

    train_inputs,train_masks, train_labels = to_tensors(train_inputs, train_masks, train_labels)
    val_inputs, val_masks,val_labels = to_tensors(val_inputs, val_masks, val_labels)
    test_inputs, test_masks, test_labels = to_tensors(test_inputs, test_masks, test_labels)


    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

    

    #####Create weighted random sampler for bootstrap
    class_counts = torch.bincount(train_labels)  # Count occurrences of each class (0 and 1)
    class_weights = 1. / class_counts.float()  # Inverse frequency weighting
    sample_weights = class_weights[train_labels]  # Assign weights to each sample


    sampler = WeightedRandomSampler(sample_weights,cfg["n_samples_dataset"], replacement=True)


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


    model = LightningCRISPertFeatures(cfg,config)


    trainer = pl.Trainer(max_epochs=cfg["epochs"],callbacks=[checkpoint_callback,early_stopping],enable_progress_bar=True,deterministic=True)
  

    trainer.fit(model, train_loader, val_loader)    

    ckpt_dir = 'checkpoints/' + cfg["output_dir"] + "/" + str(cfg["pytorch seed"]) + "/"
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    avg_ckpt_path = ckpt_files[0]


    model = LightningCRISPertFeatures.load_from_checkpoint(checkpoint_path=avg_ckpt_path, cfg=cfg,bert_config=config)

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


if __name__ == "__main__":
   
    base_cfg={
    "pytorch seed":42,
    "model_name_or_path":"CRISPert_pretrained_model/hf_model",
    "train_batch_size": 256,
    "val_batch_size": 1024,
    "test_batch_size": 256,
    "train/test_split_seed": 42,
    "learning_rate": 1e-4,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "n_samples_dataset": 2000,
    "patience": 15,
    "epochs": 100,
    "adam_epsilon":1e-8,
    "beta1":0.9,
    "beta2":0.999,
    "output_dir": "test",
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
save_path = "CRISPert_cross_cell_results_fixed_k562.csv"
df.to_csv(save_path, index=False)

print(f"Results saved to {save_path}")