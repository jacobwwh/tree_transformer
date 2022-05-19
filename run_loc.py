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
from graph_utils import create_loc_dataset,create_loc_batch,create_loc_repair_batch
from utils import dataiterator,NoamLR
from treetransformernew import TreeTransformer_localize

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--emsize", type=int, default=256, help="embedding dim")
parser.add_argument("--num_heads", type=int, default=4, help="attention heads")
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.002, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--maxepoch", type=int, default=30, help="max training epochs")
parser.add_argument("--nobar", type=boolean_string, default=False, help="disable progress bar")
args = parser.parse_args()

ops_dict={'-':0,'+':1,'*':2,'%':3,'>':4,'==':5,'or':6,'<':7,'/':8,'and':9,'>=':10,'<=':11,'!=':12,'in':13,'is':14,'is not':15,'not in':16}

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
                    label=data['correct_op']
                    dataset.append({'tree':root,'label':ops_dict[label]})
    print(len(dataset))
    return dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset=read_data_from_disk('wrongoperator/train/')
devset=read_data_from_disk('wrongoperator/dev/')
testset=read_data_from_disk('wrongoperator/test/')
vocab=pickle.load(open('wrongoperator/vocab_min5.pkl','rb'))
vocabsize=len(vocab)

#emsize = 256 # embedding dimension
nhid = args.emsize*4 # the dimension of feedforward layer
#nhead = 4 # the number of heads in the multiheadattention models
#dropout = 0.2 # the dropout value

model = TreeTransformer_localize(args.num_heads,args.emsize,nhid,vocabsize).to(device)

trainset=create_loc_dataset(trainset,vocab,withlabel=True)
devset=create_loc_dataset(devset,vocab,withlabel=True)
testset=create_loc_dataset(testset,vocab,withlabel=True)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

warmup_steps=2000
scheduler=NoamLR(optimizer,warmup_steps=warmup_steps)

maxdevacc=0
maxdevepoch=0
for epoch in range(args.maxepoch):
    print('epoch:',epoch+1)
    sys.stdout.flush()
    model.train()
    random.shuffle(devset)
    trainbar=tqdm(dataiterator(devset,batch_size=args.batch_size),disable=args.nobar)
    totalloss=0.0
    traincorrect=0
    traincorrect_locrepair=0
    for batch in trainbar:
        optimizer.zero_grad()
        inputbatch,targets,batch_maxlen,batch_numnodes,repair_labels=create_loc_repair_batch(batch,device)
        output,repair_output = model(inputbatch,batch_maxlen,batch_numnodes,repair=True)
        loc_loss = criterion(output, targets)
        repair_loss=repair_criterion(repair_output,repair_labels)
        trainbar.set_description("loc loss {}, repair loss {}".format(loc_loss.item(),repair_loss.item()))
        loss=loc_loss+repair_loss
        totalloss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        traincorrect+=(output.argmax(1) == targets).sum().item()
        traincorrect_locrepair+=((output.argmax(1) == targets)&(repair_output.argmax(1)==repair_labels)).sum().item()
    print('avg loss:',totalloss/len(trainset)*batch_size)
    print('train acc:',traincorrect/len(trainset))
    print('train acc jointly loc&repair:',traincorrect_locrepair/len(trainset))
    
    model.eval()
    with torch.no_grad():
        devbar=tqdm(dataiterator(devset,batch_size=batch_size),disable=nobar)
        devcorrect=0
        devcorrect_locrepair=0
        for batch in devbar:
            inputbatch,targets,batch_maxlen,batch_numnodes,repair_labels=create_loc_repair_batch(batch,device)
            output,repair_output = model(inputbatch,batch_maxlen,batch_numnodes,repair=True)
            devcorrect+=(output.argmax(1) == targets).sum().item()
            devcorrect_locrepair+=((output.argmax(1) == targets)&(repair_output.argmax(1)==repair_labels)).sum().item()
        devacc=devcorrect/len(devset)
        devacc_locrepair=devcorrect_locrepair/len(devset)
        print('dev acc:',devacc)
        print('dev acc jointly loc&repair:',devacc_locrepair)
        if devacc>maxdevacc:
            maxdevacc=devacc
            maxdevepoch=epoch
            print('best epoch')
            print('start testing')
            testbar=tqdm(dataiterator(testset,batch_size=batch_size),disable=nobar)
            testcorrect=0
            testcorrect_locrepair=0
            for batch in testbar:
                inputbatch,targets,batch_maxlen,batch_numnodes,repair_labels=create_loc_repair_batch(batch,device)
                output,repair_output = model(inputbatch,batch_maxlen,batch_numnodes,repair=True)
                testcorrect+=(output.argmax(1) == targets).sum().item()
                testcorrect_locrepair+=((output.argmax(1) == targets)&(repair_output.argmax(1)==repair_labels)).sum().item()
            testacc=testcorrect/len(testset)
            testacc_locrepair=testcorrect_locrepair/len(testset)
            print('test acc:',testacc)
            print('test acc jointly loc&repair:',testacc_locrepair)
    if epoch-maxdevepoch>10:
        print('early stop')
        print('best epoch:',maxdevepoch)
        quit()
