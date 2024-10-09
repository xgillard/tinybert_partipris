import numpy  as np
import pandas as pd
import torch
import torch.nn   as nn
from transformers import AutoTokenizer
from transformers import BertConfig, BertForPreTraining
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets     import Dataset
import evaluate

## GENERAL
BATCH_SZ     = 64
TRAIN_EPOCHS = 40000

# model
data      = Dataset.from_pandas(pd.read_parquet('partipris_pretraining_full.parquet'))
data      = data.train_test_split(train_size=0.8, shuffle=True)
tokenizer = AutoTokenizer.from_pretrained("xaviergillard/parti-pris-v2")
model     = BertForPreTraining.from_pretrained("xaviergillard/parti-pris-v2", torch_dtype=torch.bfloat16) 

# metrics
loss     = nn.CrossEntropyLoss()
accuracy = evaluate.load("accuracy")
def ignoring_dummy(preds, labels, dummy=-100):
    yhat = []
    y    = []
    labels = labels.reshape((-1,))
    preds  = preds.reshape((labels.shape[0], -1))    
    for i,label in enumerate(labels):
        if label == dummy:
            continue
        else:
            y.append(label)
            yhat.append(preds[i].argmax())
    yhat = np.array(yhat)
    y    = np.array(y)
    return (yhat, y)
    
def compute_metrics(eval):
    y_mlm, y_nsp = eval.label_ids
    h_mlm, h_nsp = eval.predictions
    #
    y_mlm = torch.tensor(y_mlm.reshape((-1,)))
    h_mlm = torch.tensor(h_mlm.reshape((y_mlm.shape[0], -1)))
    l_mlm = loss(h_mlm, y_mlm)
    
    y_nsp = torch.tensor(y_nsp.reshape((-1,)))
    h_nsp = torch.tensor(h_nsp.reshape((y_nsp.shape[0], -1)))
    l_nsp = loss(h_nsp, y_nsp)
    #
    h_mlm, y_mlm = ignoring_dummy(h_mlm, y_mlm, dummy=-100)
    a_mlm = accuracy.compute(predictions=h_mlm, references=y_mlm)
    a_nsp = accuracy.compute(predictions=h_nsp.argmax(axis=-1), references=y_nsp)
    #
    return {
        'mlm_accuracy': a_mlm['accuracy'], 
        'nsp_accuracy': a_nsp['accuracy'], 
        'mlm_loss': l_mlm, 
        'nsp_loss': l_nsp, 
        'tot_loss': l_mlm + l_nsp 
    }

# training
collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer)
args      = TrainingArguments(
    num_train_epochs            = TRAIN_EPOCHS,
    per_device_train_batch_size = BATCH_SZ,
    #
    output_dir                  = './checkpoints', 
    overwrite_output_dir        = True,
    save_strategy               = "epoch", 
    save_total_limit            = 2,
    #
    eval_strategy               = "epoch",
    #
    gradient_accumulation_steps = 100,
    bf16                        = True,
    #
    push_to_hub                 = True,
    hub_model_id                = "xaviergillard/parti-pris-v2",
    hub_strategy                = "every_save",
    hub_token                   = "FIXME")

trainer = Trainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = data['train'], 
    eval_dataset    = data['test'],
    args            = args,
    data_collator   = collator,
    compute_metrics = compute_metrics
) 

trainer.train()

# the end
model.save_pretrained("pretrained/partipris")
model.push_to_hub("xaviergillard/parti-pris-v2")
