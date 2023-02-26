import tensorflow as tf
from datasets import load_dataset, load_metric
from transformers import BertConfig
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import numpy as np
#for pytorch
import torch
import torch.nn as nn
device = torch.device("cuda")
import argparse
#for BERT
import transformers
from transformers import AutoModel
from transformers import AdamW

# loading self trained word level tokenizer that assumes tokenized input
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import sys
import torch.nn.functional as F
from tqdm import tqdm
import os

def preprocess_encode(examples):
    sentence1_key, sentence2_key = task_to_keys[actual_task]
    args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
    return tokenizer(*args, padding='max_length', max_length=128, truncation=True)
    
def createSeqMaskLbl(split):
    tokens_split = preprocess_encode(dataset[split])
    split_seq = torch.tensor(tokens_split['input_ids'])
    split_mask = torch.tensor(tokens_split['attention_mask'])
    if task == "STSB":
        split_y = torch.tensor([sen['label']/5 for sen in dataset[split]])
    else:
        split_y = torch.tensor([round(sen['label']) for sen in dataset[split]])
    
    return split_seq,split_mask,split_y
 
def createDataLoader(split):
    split_seq,split_mask,split_y=createSeqMaskLbl(split)
    batch_size = 32

    # wrap tensors
    split_data = TensorDataset(split_seq, split_mask, split_y)

    # sampler for sampling the data during training
    split_sampler = RandomSampler(split_data)

    # dataLoader for train set
    return DataLoader(split_data, sampler=split_sampler, batch_size=batch_size)

def getPTmodel(model):
    model = model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

def getCleanModel(cleanModel,model):
    model.load_state_dict(torch.load(cleanModel))
    model=model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'embeddings' in name:
            continue
        if 'layer' in name and int(name.split('.')[3])<6:
            continue
        param.requires_grad = True
    return model
    
def getModelOpt():
    cleanModel = f'models/DeBERTa_FT_{task}.mdl'
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base"+modelKey, num_labels=num_labels)
    if os.path.exists(cleanModel):
        model = getCleanModel(cleanModel,model)
        isFT = True
    else:
        model = getPTmodel(model)
        isFT = False


    optimizer = AdamW(model.parameters(), lr = leraningRate) 
    return model,optimizer, isFT

def getAdvLoss(model, args, method, x_data,attention_mask, model_out=None, y_data=None):
    adv = createAdv(model, args, method, x_data,attention_mask, model_out=model_out, y_data=y_data)
    return getLoss(model, method, adv,attention_mask, model_out=model_out, y_data=y_data, isAdvCreate = False) 
    

def createAdv(model, args, method, x_data,attention_mask, model_out=None, y_data=None):
    X_adv = x_data.clone().detach()
    noise = torch.normal(0, args['noise_epsilon'], size=X_adv.shape).to(device)
    X_adv += noise
    
    for i in range(args['pdg_step']):
        X_adv = X_adv.requires_grad_(True)
        loss = getLoss(model, method, X_adv,attention_mask, model_out=model_out, y_data=y_data, isAdvCreate = True) 
        
        grad_X_adv = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv.detach().requires_grad_(False)
        
        X_adv += args['epsilon']* grad_X_adv.sign()
        eta = torch.clamp(X_adv-x_data, min=-1*args['clipLmt'], max=args['clipLmt'])
        X_adv = x_data.clone().detach() + eta
        
    X_adv.detach_()
    return X_adv

# In[92]:
def getLoss(model, method, X_adv, attention_mask,model_out=None, y_data=None, isAdvCreate = False):
    if method == 'AT':
        return getALSTLoss(model,X_adv,attention_mask,y_data)
    if method == 'A3T':
        return getProposedLoss(model,X_adv,attention_mask,model_out, y_data, isAdvCreate)

    raise NotImplementedError("ADV loss only be one of the AT or A3T")

def getYpred(model_out, y_data, isAdvCreate):
    if isAdvCreate:
        if task == "STSB":
            y_pred = model_out.detach().cpu().numpy()
        else:
            y_pred = np.argmax(model_out.detach().cpu().numpy(), axis = 1)
        return torch.tensor(y_pred).reshape(y_data.shape).type(y_data.type()).to(y_data.device)
        
    else:
        return y_data
    
    
def getALSTLoss(model,X_adv,attention_mask,y_data):
    adv_out = model(inputs_embeds=X_adv,labels=y_data,attention_mask=attention_mask)
    return adv_out.loss  

def getProposedLoss(model, X_adv,attention_mask,model_out, y_data, isAdvCreate):    
    y_pred = getYpred(model_out, y_data, isAdvCreate) 
    adv_out = model(inputs_embeds=X_adv,labels=y_pred,attention_mask=attention_mask)
    return adv_out.loss
  

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "STSB":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def getAcc(model,x,y,attention_mask):
    model_out = model(inputs_embeds=x,attention_mask=attention_mask)
    met= compute_metrics((model_out.logits.detach().cpu().numpy(),y.detach().cpu().numpy()))
    return np.array(list(met.values()))

def getAttckAcc(model,x_test,y_test,attention_mask,args):
    X_adv=createAdv(model, args, 'AT', x_test,attention_mask, y_data=y_test)
    return getAcc(model,X_adv,y_test,attention_mask)

def createArgs(noise_epsilon,pdg_step, epsilon, clipLmt):
    args={}
    args['noise_epsilon'] = noise_epsilon
    args['pdg_step'] = pdg_step
    args['epsilon'] = epsilon
    args['clipLmt'] = clipLmt
    return args

def runMethod(method,train_dataloader, attck_args):

    model,optim, isFT = getModelOpt()

    if isFT:
        #first three epoch is a clean training, if isFT is true, this mean clean model is loaded
        trainingEpochN = trainingEpochNum-3
    else:
        if method!= 'clean':
            #there is no clean model, first 3 epochs fine tuning is needed
            raise NotImplementedError(f"There is no clean model for adv training, first run the code with clean parameters like: python glue.py {task} clean")
        trainingEpochN = trainingEpochNum
        
    for epoch in tqdm(range(trainingEpochN)):
        if epoch==3 and method == 'clean' and (not isFT):
            torch.save(model.state_dict(), f'models/DeBERTa_FT_{task}.mdl')
            
        for step,batch in enumerate(train_dataloader):
            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, attention_mask, labels = batch
            #to calculate grad, we need token embeddings
            inputs_embeds=model.deberta.get_input_embeddings().weight[sent_id].clone()
        
            optim.zero_grad()
            
            #get clean loss and prediction
            model_outs = model(sent_id,attention_mask=attention_mask,labels=labels)
            loss1=model_outs.loss
            model_out=model_outs.logits

            if method == 'clean': #first 3 epochs clean
                loss = loss1
            else:
                #if it is adversarial trainin, get adv loss based on adv method
                loss2 = getAdvLoss(model,attck_args, method, inputs_embeds,attention_mask,
                                   model_out=model_out, y_data=labels)
                loss = loss1 + loss2
            loss.backward()
            optim.step()
       
    
    return model


def testModel(model,test_dataloader):
    all_acc=[]
    all_attck_acc=[]
    for step,batch in enumerate(test_dataloader):
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch
        
        inputs_embeds=model.deberta.get_input_embeddings().weight[sent_id].clone()
        
        #get the clean acc
        all_acc.append(getAcc(model,inputs_embeds,labels,mask))

        pdg_step=3
        attck_acc=[]
        # we can test different attack parameters, however this is not a realistic scenerio
        for epsilon_tmp in [0.005]:
            for noise_epsilon_tmp in [0.005]:
                for clipLmt_tmp in [0.01]:
                    attack_args = createArgs(noise_epsilon_tmp,3, epsilon_tmp, clipLmt_tmp)
                    acc = getAttckAcc(model,inputs_embeds,labels,mask,attack_args)
                    attck_acc.append(acc)
                    
        all_attck_acc.append(attck_acc)
        
    acc=np.mean(np.array(all_acc),axis=0)
    attck_acc=np.mean(np.array(all_attck_acc),axis=0)
    print(acc,attck_acc)
    
    
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
task2lr={
    "cola": (2e-5,6),
    "mrpc": (2e-5,6),
    "qnli": (2e-5,6),
    "qqp": (2e-5,6),
    "rte": (2e-5,6),
    "sst2": (2e-5,6),
    "stsb": (2e-5,6),
    "mnli": (2e-5,6),
}
epsilon = 0.005
noise_epsilon = 0.005
clipLmt = 0.01
pdg_step = 3

if __name__ == "__main__":
    
    task = sys.argv[1]
    methodName = sys.argv[2]

    print('--------------')
    print(task)
    actual_task = task.lower().replace('-','')

    num_labels=2
    if actual_task == 'mnli':
        additonalKey='_matched'
        modelKey='-mnli'
        num_labels=3
    else:
        additonalKey=''
        modelKey=''
        
    if actual_task == 'stsb':
        num_labels=1
        
    #get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base"+modelKey)
    #default learning rate and epoch counts
    leraningRate,trainingEpochNum = task2lr[actual_task]
    
    #load dataset and metrics
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    
    #create loaders
    train_dataloader = createDataLoader('train')  
    val_dataloader = createDataLoader('validation'+additonalKey)
    test_dataloader = createDataLoader('test'+additonalKey)

    
    if not os.path.exists('models'):
        os.mkdir('models')
    modelFile=f'models/{methodName}_{epsilon}_{noise_epsilon}_{clipLmt}_DeBERTa-{task}.mdl'
    
    #if the model is exist, no need for training
    if os.path.exists(modelFile):
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base"+modelKey, num_labels=num_labels)
        model.load_state_dict(torch.load(modelFile))
        model=model.to(device)
    else:
        attck_args = createArgs(noise_epsilon,pdg_step, epsilon, clipLmt)
        model = runMethod(methodName,train_dataloader,attck_args)
        torch.save(model.state_dict(), modelFile)
        
    #test the model
    model.eval()
    testModel(model,val_dataloader)


