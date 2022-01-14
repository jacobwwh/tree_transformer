#run experiments for codenet

from preprocess_codenet import get_spt_dataset
import sys
import argparse
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
from utils import dataiterator,NoamLR
from treetransformernew import TreeTransformer_typeandtoken

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='java250, help="dataset name")
parser.add_argument("--emsize", type=int, default=256, help="embedding dim")
parser.add_argument("--num_heads", type=int, default=4, help="attention heads")
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.002, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--maxepoch", type=int, default=500, help="max training epochs")
parser.add_argument("--nobar", type=boolean_string, default=False, help="disable progress bar")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_classes={'java250':250,'python800':800,'c++1000':1000,'c++1400':1400}
nclasses=num_classes[args.dataset_name]

trainset,devset,testset,token_vocabsize,type_vocabsize=get_spt_dataset(bidirection=False)

#emsize = 256 # embedding dimension
nhid = args.emsize*4 # the dimension of feedforward layer
#nhead = 4 # the number of heads in the multiheadattention models
#dropout = 0.2 # the dropout value

print('embedsize:',emsize,'hidden:',nhid,'heads:',nhead)

#model = TreeTransformer_typeandtoken(nhead,emsize,dk,dv,nhid,nclasses,token_vocabsize,type_vocabsize,dropout=dropout,top_down=False).to(device)                   
model = TreeTransformer_typeandtoken(args.num_heads,args.emsize,nhid,nclasses,token_vocabsize,type_vocabsize,dropout=args.dropout,top_down=True).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

warmup_steps=2000
scheduler=NoamLR(optimizer,warmup_steps=warmup_steps)
#batch_size=256
#print(batch_size,lr,warmup_steps)


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

print('max epoch:',args.maxepoch)
maxdevacc=0
maxdevepoch=0
for epoch in range(maxepoch):
    print('epoch:',epoch+1)
    sys.stdout.flush()
    model.train()
    random.shuffle(trainset)
    trainbar=tqdm(dataiterator(trainset,batch_size=args.batch_size),disable=args.nobar)
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
        devbar=tqdm(dataiterator(devset,batch_size=args.batch_size),disable=args.nobar)
        devtotal=len(devset)
        devcorrect=0
        for batch in devbar:
            inputbatch,targets,root_ids=create_graph_batch(batch,device)
            output = model(inputbatch,root_ids=root_ids)
            devcorrect+=(output.argmax(1) == targets).sum().item()
        print('devacc:',devcorrect/devtotal)
        testbar=tqdm(dataiterator(testset,batch_size=args.batch_size), disable=args.nobar)
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
