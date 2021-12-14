#run experiments for codenet

from preprocess_codenet import get_spt_dataset
import sys
import random
import pickle
import anytree
from anytree import AnyNode, RenderTree
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
import torchtext
from torchtext.data.utils import get_tokenizer
from graph_utils import create_graph_batch
from utils import dataiterator
from treetransformernew import TreeTransformer_typeandtoken
from treetransformer_acl import TreeTransformer_typeandtoken as acl2019classifier
from treelstm import TreeLSTM_typeandtoken
from tbcnn import TBCNN_typeandtoken
from model.gnn_baseline import GNN_typeandtoken
from treecaps import TreeCaps_typeandtoken
from model.radam import RAdam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_name='java250'
print(dataset_name)
model_name='tree-transformer'
print(model_name)

num_classes={'java250':250,'python800':800,'c++1000':1000,'c++1400':1400}
nclasses=num_classes[dataset_name]

if model_name in ['gcn','gin','ggnn']:
    trainset,devset,testset,token_vocabsize,type_vocabsize=get_spt_dataset(bidirection=True)
elif model_name in ['ggnn-typed']:
    trainset,devset,testset,token_vocabsize,type_vocabsize=get_spt_dataset(edgetype=True)
else:
    trainset,devset,testset,token_vocabsize,type_vocabsize=get_spt_dataset(bidirection=False)

emsize = 256 # embedding dimension
nhid = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
dk=32 #key dimension
dv=32 #value dimension
nlayers = 1 # the number of Transformer Encoder Layers
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

print('embedsize:',emsize,'hidden:',nhid,'key:',dk,'value:',dv,'layers:',nlayers,'heads:',nhead)

if model_name=='tree-transformer':
    model = TreeTransformer_typeandtoken(nhead,emsize,dk,dv,nhid,nclasses,token_vocabsize,type_vocabsize,dropout=dropout,top_down=True).to(device)
    #model=acl2019classifier(nhead,emsize,dk,dv,nhid,nclasses,token_vocabsize,type_vocabsize,dropout=dropout).to(device)
elif model_name=='tree-lstm':
    model = TreeLSTM_typeandtoken(emsize,emsize,dropout,nclasses,token_vocabsize,type_vocabsize).to(device)
elif model_name=='gcn':
    model=GNN_typeandtoken(emsize,nclasses,5,token_vocabsize,type_vocabsize,dropout=dropout,model='gcn').to(device)
elif model_name=='gin':
    model=GNN_typeandtoken(emsize,nclasses,5,token_vocabsize,type_vocabsize,dropout=dropout,model='gin').to(device)
elif model_name=='ggnn' or model_name=='ggnn-typed':
    model=GNN_typeandtoken(emsize,nclasses,5,token_vocabsize,type_vocabsize,dropout=dropout,model='ggnn').to(device)
elif model_name=='tbcnn':
    model=TBCNN_typeandtoken(emsize,emsize,nclasses,token_vocabsize,type_vocabsize,num_layers=8).to(device)
elif model_name=='treecaps':
    model=TreeCaps_typeandtoken(emsize,emsize,nclasses,token_vocabsize,type_vocabsize,num_layers=8).to(device)

criterion = nn.CrossEntropyLoss()
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
batch_size=256
print(batch_size,lr,warmup_steps)


def createdatabatch_spt(batch,device):
    graphs,labels=list(zip(*batch))
    root_ids=[]
    rootid=0
    for graph in graphs:
        root_ids.append(rootid)
        rootid+=graph.number_of_nodes()
    labels=torch.tensor(labels,dtype=torch.long).to(device)
    batched_graph=dgl.batch(graphs).to(device)
    return batched_graph,labels,root_ids

maxepoch=500
print('max epoch:',maxepoch)
nobar=True
maxdevacc=0
maxdevepoch=0
for epoch in range(maxepoch):
    print('epoch:',epoch+1)
    sys.stdout.flush()
    model.train()
    random.shuffle(trainset)
    trainbar=tqdm(dataiterator(trainset,batch_size=batch_size),disable=nobar)
    totalloss=0.0
    traincorrect=0
    for batch in trainbar:
        optimizer.zero_grad()
        inputbatch,targets,root_ids=create_graph_batch(batch,device)
        output = model(inputbatch,root_ids=root_ids)
        loss = criterion(output, targets)
        trainbar.set_description("loss {}".format(loss.item()))
        totalloss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        traincorrect+=(output.argmax(1) == targets).sum().item()
    print('avg loss:',totalloss/len(trainset)*batch_size)
    print('train acc:',traincorrect/len(trainset))
    
    model.eval()
    with torch.no_grad():
        devbar=tqdm(dataiterator(devset,batch_size=batch_size),disable=nobar)
        devtotal=len(devset)
        devcorrect=0
        for batch in devbar:
            inputbatch,targets,root_ids=create_graph_batch(batch,device)
            output = model(inputbatch,root_ids=root_ids)
            devcorrect+=(output.argmax(1) == targets).sum().item()
        print('devacc:',devcorrect/devtotal)
        testbar=tqdm(dataiterator(testset,batch_size=batch_size), disable=nobar)
        testtotal=len(testset)
        testcorrect=0
        for batch in testbar:
            inputbatch,targets,root_ids=create_graph_batch(batch,device)
            output = model(inputbatch,root_ids=root_ids)
            testcorrect+=(output.argmax(1) == targets).sum().item()
        print('testacc:',testcorrect/testtotal)
    if devcorrect/devtotal>=maxdevacc:
        maxdevacc=devcorrect/devtotal
        maxdevepoch=epoch
        print('best epoch')
    if epoch-maxdevepoch>30:
        print('early stop')
        print('best epoch:',maxdevepoch)
        quit()
