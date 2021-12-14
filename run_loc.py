#run the wrong operator localization experiment

import os
import sys
import argparse
import random
import pickle
import json
import anytree
from anytree import AnyNode, RenderTree
from anytree.importer import JsonImporter
from anytree.search import findall
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from graph_utils import create_loc_dataset,create_loc_batch
from decoder.enc_dec_utils import Tree2SeqDataset,collect_fn,label_smoothing_loss,calculate
from utils import dataiterator
from treetransformernew import TreeTransformer_localize
from model.gnn_baseline import GNN_localize

def read_data_from_disk(path):
    files = os.listdir(path)
    importer = JsonImporter()
    dataset=[]
    overlencount=0
    for file in files:
        file_path = os.path.join(path, file)
        if file_path.endswith('.jsonl'):
            print(file_path)
            with open(file_path) as f:
                lines=f.readlines()
                for line in lines:
                    data=json.loads(line)
                    root = importer.import_(data['tree'])
                    overlength=False
                    cursor=root
                    while cursor.children:
                        cursor=cursor.children[-1]
                        if cursor.index>2000:
                            overlength=True
                            overlencount+=1
                            #print(cursor)
                            break    
                    #print(data['label'])
                    opnodes=findall(root, lambda node: node.is_operator==True)
                    if overlength==False and len(opnodes)>1:
                        dataset.append({'tree':root})
    print(len(dataset))
    return dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset=read_data_from_disk('wrongoperator/train/')
devset=read_data_from_disk('wrongoperator/dev/')
testset=read_data_from_disk('wrongoperator/test/')
vocab=pickle.load(open('wrongoperator/vocab_min5.pkl','rb'))
vocabsize=len(vocab)

emsize = 256 # embedding dimension
nhid = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
dk=32 #key dimension
dv=32 #value dimension
nlayers = 1 # the number of Transformer Encoder Layers
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

print('embedsize:',emsize,'hidden:',nhid,'key:',dk,'value:',dv,'layers:',nlayers,'heads:',nhead)

model = TreeTransformer_localize(nhead,emsize,emsize,vocabsize).to(device)

trainset=create_loc_dataset(trainset,vocab)
devset=create_loc_dataset(devset,vocab)
testset=create_loc_dataset(testset,vocab)

criterion = nn.NLLLoss()
lr = 5.0 # learning rate
lr=0.002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

from torch.optim.lr_scheduler import _LRScheduler
class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        #scale=min(last_epoch/self.warmup_steps, 1)
        return [base_lr * scale for base_lr in self.base_lrs]

warmup_steps=2000
scheduler=NoamLR(optimizer,warmup_steps=warmup_steps)
batch_size=64
print(batch_size,lr,warmup_steps)


nobar=False
maxdevacc=0
maxdevepoch=0
for epoch in range(300):
    print('epoch:',epoch+1)
    sys.stdout.flush()
    model.train()
    random.shuffle(devset)
    trainbar=tqdm(dataiterator(devset,batch_size=batch_size),disable=nobar)
    totalloss=0.0
    traincorrect=0
    for batch in trainbar:
        optimizer.zero_grad()
        inputbatch,targets,batch_maxlen,batch_numnodes=create_loc_batch(batch,device)
        output = model(inputbatch,batch_maxlen,batch_numnodes)
        #print(output.size(),targets.size())
        loss = criterion(output, targets)
        trainbar.set_description("loss {}".format(loss.item()))
        totalloss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        traincorrect+=(output.argmax(1) == targets).sum().item()
    print('avg loss:',totalloss/len(devset)*batch_size)
    print('train acc:',traincorrect/len(devset))
