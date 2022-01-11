from preprocess_poj import createdata as createdata_pyc
from preprocess_poj import read_data_from_disk
import sys
import random
import pickle
import anytree
from anytree import AnyNode, RenderTree
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator
from graph_utils import create_graph_dataset, create_graph_batch
from utils import dataiterator
from treetransformernew import TreeTransformerClassifier as newclassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#dataset_name='devign_ast_camelsnake'
#print(dataset_name)

trainsamples=read_data_from_disk('poj104_pyc/traindata_712.jsonl')
devsamples=read_data_from_disk('poj104_pyc/devdata_712.jsonl')
testsamples=read_data_from_disk('poj104_pyc/testdata_712.jsonl')    

vocab=pickle.load(open('poj104_pyc/typevocab.pkl','rb'))
nclasses=104


print(len(trainsamples),len(devsamples),len(testsamples))
print(len(vocab))
#modelname='tree-transformer'

trainset=create_graph_dataset(trainsamples,vocab)
devset=create_graph_dataset(devsamples,vocab)
testset=create_graph_dataset(testsamples,vocab)

ntokens = len(vocab.stoi) # the size of vocabulary
emsize = 128 # embedding dimension
nhid = emsize*4 # the dimension of the feedforward network model in nn.TransformerEncoder
dk=32 #key dimension
dv=32 #value dimension
nlayers = 1 # the number of Transformer Encoder Layers
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

print('embedsize:',emsize,'hidden:',nhid,'key:',dk,'value:',dv,'layers:',nlayers,'heads:',nhead)

model = newclassifier(nhead,emsize,dk,dv,nhid,nclasses,ntokens,dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr=0.002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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

nobar=False
maxdevacc=0
maxdevepoch=0
for epoch in range(500):
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
    print('avg loss:',totalloss/len(trainsamples)*batch_size)
    print('train acc:',traincorrect/len(trainsamples))
    
    model.eval()
    with torch.no_grad():
        devbar=tqdm(dataiterator(devset,batch_size=batch_size),disable=nobar)
        devtotal=len(devsamples)
        devcorrect=0
        for batch in devbar:
            inputbatch,targets,root_ids=create_graph_batch(batch,device)
            output = model(inputbatch,root_ids=root_ids)
            devcorrect+=(output.argmax(1) == targets).sum().item()
        print('devacc:',devcorrect/devtotal)
        testbar=tqdm(dataiterator(testset,batch_size=batch_size), disable=nobar)
        testtotal=len(testsamples)
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
    if epoch-maxdevepoch>50:
        print('early stop')
        print('best epoch:',maxdevepoch)
        quit()
