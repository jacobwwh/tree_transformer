from preprocess_poj import read_data_from_disk
import sys
import random
import pickle
import anytree
from anytree import AnyNode, RenderTree
from tqdm import tqdm
import torch
import torch.nn as nn
from graph_utils import create_graph_dataset, create_graph_batch
from utils import dataiterator,NoamLR
from treetransformernew import TreeTransformerClassifier as newclassifier

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--emsize", type=int, default=128, help="embedding dim")
parser.add_argument("--num_heads", type=int, default=4, help="attention heads")
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.002, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--maxepoch", type=int, default=500, help="max training epochs")
parser.add_argument("--nobar", type=boolean_string, default=False, help="disable progress bar")
args = parser.parse_args()

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
vocabsize=len(vocab)

#emsize = 256 # embedding dimension
nhid = args.emsize*4 # the dimension of feedforward layer
#nhead = 4 # the number of heads in the multiheadattention models
#dropout = 0.2 # the dropout value


model = newclassifier(nhead,emsize,nhid,nclasses,vocabsize,dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

warmup_steps=2000
scheduler=NoamLR(optimizer,warmup_steps=warmup_steps)

maxdevacc=0
maxdevepoch=0
for epoch in range(args.maxepoch):
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
    print('avg loss:',totalloss/len(trainsamples)*batch_size)
    print('train acc:',traincorrect/len(trainsamples))
    
    model.eval()
    with torch.no_grad():
        devbar=tqdm(dataiterator(devset,batch_size=args.batch_size),disable=args.nobar)
        devtotal=len(devsamples)
        devcorrect=0
        for batch in devbar:
            inputbatch,targets,root_ids=create_graph_batch(batch,device)
            output = model(inputbatch,root_ids=root_ids)
            devcorrect+=(output.argmax(1) == targets).sum().item()
        print('devacc:',devcorrect/devtotal)
        
        testbar=tqdm(dataiterator(testset,batch_size=args.batch_size), disable=args.nobar)
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
