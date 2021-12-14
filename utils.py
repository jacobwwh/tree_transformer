from numpy.lib.histograms import _ravel_and_check_weights
import anytree
from anytree import AnyNode, RenderTree
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter, defaultdict
from torchtext.vocab import Vocab, build_vocab_from_iterator
from models.utils.misc import count_file_lines
from models.utils.misc import generate_relative_positions_matrix

def getseq(tree):
    """get the depth-first traversal of an anytree ast.
    generate relation pairs"""
    dfsseq=[]
    childdict=defaultdict(list)
    leaf2rootpaths=[]
    def traverse(node):
        dfsseq.append(node.name)
        childdict[node.index]=[]
        if not node.children:
            leaf2rootpaths.append(node.rootpath) #add path to root
        for child in node.children:
            childdict[node.index].append(child.index)
            traverse(child)
    traverse(tree)
    return dfsseq, childdict,leaf2rootpaths

def build_vocab_from_iterator(iterator, num_lines=None,vocab_size=None):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        num_lines: The expected number of elements returned by the iterator.
            (Default: None)
            Optionally, if known, the expected number of elements can be passed to
            this factory function for improved progress reporting.
    """
    counter = Counter()
    with tqdm(unit_scale=0, unit='lines', total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    if vocab_size:
        word_vocab = Vocab(counter,max_size=vocab_size)
    else:
        word_vocab = Vocab(counter)
    return word_vocab

def get_batch(samples,vocab,maxlen=None):
    """generate a batch of input for basic transformer model."""
    inputs,targets=list(zip(*samples))
    inputbatch=[]
    targetbatch=torch.tensor(targets,dtype=torch.long)
    batch_maxlen=0
    data_lengths=[]
    for data in inputs:
        data=getseq(data)
        if len(data)>maxlen:
            data=data[:maxlen]
        if len(data)>batch_maxlen:
            batch_maxlen= len(data)
        data_lengths.append(len(data))
        inputtensor=torch.tensor([vocab[token] for token in data],dtype=torch.long)
        inputbatch.append(inputtensor)
    inputbatch=pad_sequence(inputbatch,batch_first=True,padding_value=vocab['<pad>'])
    padding_mask=torch.zeros(len(samples),batch_maxlen)
    for i in range(len(samples)):
        padding_mask[i, data_lengths[i]:] = 1
    padding_mask=padding_mask.bool()
    return inputbatch,targetbatch,padding_mask

def get_batch_withlen(samples,vocab,maxlen=None,max_relpos=20,neg_relpos=False):
    """generate a batch of input for basic transformer model."""
    inputs,targets=list(zip(*samples))
    inputbatch=[]
    targetbatch=torch.tensor(targets,dtype=torch.long)
    batch_maxlen=0
    data_lengths=[]
    astmasks=[] #ast parent-child mask
    sibmasks=[] #sibling mask
    ancmasks=[]
    childRelPos=[] # parent-child relative position matrix
    sibRelPos=[] # sibling relative position matrix
    ancRelPos=[]
    childdicts=[]
    for data in inputs:
        data,childdict,leaf2rootpaths=getseq(data)
        childdicts.append(childdict)
        if len(data)>maxlen:
            data=data[:maxlen]
        if len(data)>batch_maxlen:
            batch_maxlen= len(data)
        data_lengths.append(len(data))
        inputtensor=torch.tensor([vocab[token] for token in data],dtype=torch.long)
        inputbatch.append(inputtensor)
    inputbatch=pad_sequence(inputbatch,batch_first=True,padding_value=vocab['<pad>'])
    lengths=torch.tensor(data_lengths,dtype=torch.long)
    for childdict in childdicts:
        astmask=torch.zeros(batch_maxlen,batch_maxlen)
        childpos=torch.zeros(batch_maxlen,batch_maxlen).long()
        sibmask=torch.zeros(batch_maxlen,batch_maxlen)
        sibpos=torch.zeros(batch_maxlen,batch_maxlen).long()
        ancestormask=torch.zeros(batch_maxlen,batch_maxlen)
        ancestorpos=torch.zeros(batch_maxlen,batch_maxlen).long()
        for node,children in childdict.items():
            if node<batch_maxlen:
                astmask[node][node]=1 #allow self attention
                for i in range(len(children)):
                    if children[i]<batch_maxlen:
                        astmask[node][children[i]]=1
                        astmask[children[i]][node]=1
                        childpos[node][children[i]]=i+1 #only from parent to children
                        childpos[children[i]][node]=-(i+1) 
                    #for j in range(len(children)):
                        #if children[i]<batch_maxlen and children[j]<batch_maxlen:
                            #sibmask[children[i]][children[j]]=1
                            #sibpos[children[i]][children[j]]=i-j
                if len(children)>0 and children[-1]>=batch_maxlen:
                    children=[id for id in children if id<batch_maxlen]
                if len(children)>0:
                    sibmatrix=generate_relative_positions_matrix(length=len(children),max_relative_positions=max_relpos,use_neg_dist=neg_relpos)
                    mesh = np.ix_(*[children,children]) #get the subtensor for siblings
                    sibpos[mesh]=sibmatrix
                    sibmask[mesh]=1
        for path in leaf2rootpaths:
            if path[-1]>=batch_maxlen:
                path=[id for id in path if id<batch_maxlen]
            ancestormatrix=generate_relative_positions_matrix(length=len(path),max_relative_positions=20,use_neg_dist=False)
            mesh = np.ix_(*[path,path]) #get the subtensor for ancestors
            ancestormask[mesh]=1
            ancestorpos[mesh]=ancestormatrix
        astmasks.append(astmask)
        sibmasks.append(sibmask)
        ancmasks.append(ancestormask)
        childRelPos.append(childpos)
        sibRelPos.append(sibpos)
        ancRelPos.append(ancestorpos)
    astmasks=torch.stack(astmasks,dim=0) # size: (batch_size, inputlen, inputlen)
    sibmasks=torch.stack(sibmasks,dim=0) # size: (batch_size, inputlen, inputlen)
    ancmasks=torch.stack(ancmasks,dim=0)
    childRelPos=torch.stack(childRelPos,dim=0).long() # size: (batch_size, inputlen, inputlen)
    sibRelPos=torch.stack(sibRelPos,dim=0).long() # size: (batch_size, inputlen, inputlen)
    ancRelPos=torch.stack(ancRelPos,dim=0).long()
    childRelPos=childRelPos.clamp(min=-max_relpos,max=max_relpos)
    childRelPos+=max_relpos #use negative relpos
    #sibRelPos=sibRelPos.clamp(min=-max_relpos,max=max_relpos)
    return inputbatch,targetbatch,lengths,[astmasks,sibmasks,ancmasks],[childRelPos,sibRelPos,ancRelPos]

class dataiterator(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.end_of_data = False
        self.start_position = 0
        self.data=data
        self.end = len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.end_of_data:
            raise StopIteration
        ss = self.start_position
        ee = self.start_position + self.batch_size
        self.start_position += self.batch_size
        if ee >= self.end:
            self.end_of_data = True
            #ss = self.end - self.batch_size
            ee=self.end
        return self.data[ss:ee]

def create_dataset(samples,maxlen,vocab,neg_dist=False):
    """create dataset for transformer"""
    print('generating dataset...')
    dataset=[]
    for sample in samples:
        data=[]
        tree,label=sample
        #print(label)
        tokenseq,childdict,leaf2rootpaths=getseq(tree)
        if len(tokenseq)>maxlen:
            tokenseq=tokenseq[:maxlen]
        inputlen=len(tokenseq)
        inputtensor=torch.tensor([vocab[token] for token in tokenseq],dtype=torch.long)
        '''astmask=torch.zeros(inputlen,inputlen)
        childpos=torch.zeros(inputlen,inputlen).long()
        sibmask=torch.zeros(inputlen,inputlen)
        sibpos=torch.zeros(inputlen,inputlen).long()
        ancestormask=torch.zeros(inputlen,inputlen)
        ancestorpos=torch.zeros(inputlen,inputlen).long()'''
        astmask=np.zeros((inputlen,inputlen))
        childpos=np.zeros((inputlen,inputlen),dtype=np.int)
        sibmask=np.zeros((inputlen,inputlen))
        sibpos=np.zeros((inputlen,inputlen),dtype=np.int)
        ancestormask=np.zeros((inputlen,inputlen))
        ancestorpos=np.zeros((inputlen,inputlen),dtype=np.int)
        for node,children in childdict.items():
            if node<inputlen:
                astmask[node][node]=1 #allow self attention
                for i in range(len(children)):
                    if children[i]<inputlen:
                        astmask[node][children[i]]=1
                        astmask[children[i]][node]=1
                        childpos[node][children[i]]=i+1 #only from parent to children
                        childpos[children[i]][node]=i+1
                    #for j in range(len(children)):
                        #if children[i]<batch_maxlen and children[j]<batch_maxlen:
                            #sibmask[children[i]][children[j]]=1
                            #sibpos[children[i]][children[j]]=i-j
                if len(children)>0 and children[-1]>=inputlen:
                    children=[id for id in children if id<inputlen]
                if len(children)>0:
                    sibmatrix=generate_relative_positions_matrix(length=len(children),max_relative_positions=20,use_neg_dist=False).numpy()
                    mesh = np.ix_(*[children,children]) #get the subtensor for siblings
                    sibpos[mesh]=sibmatrix
                    sibmask[mesh]=1
        for path in leaf2rootpaths:
            if path[-1]>=inputlen:
                path=[id for id in path if id<inputlen]
            ancestormatrix=generate_relative_positions_matrix(length=len(path),max_relative_positions=20,use_neg_dist=False).numpy()
            mesh = np.ix_(*[path,path]) #get the subtensor for ancestors
            ancestormask[mesh]=1
            ancestorpos[mesh]=ancestormatrix
        masks=[astmask,sibmask,ancestormask]
        positions=[childpos,sibpos,ancestorpos]
        data=[inputtensor,inputlen,masks,positions,label]
        #dataset.append(data)
    return dataset

def padmasks(tensors,maxlen,device):
    for i in range(len(tensors)):
        tensors[i]=torch.tensor(tensors[i])
        datalen=tensors[i].size()[0]
        tensors[i]=tensors[i].to(device)
        tensors[i]=F.pad(tensors[i],pad=(0,maxlen-datalen,0,maxlen-datalen),value=0)
    tensors=torch.stack(tensors,dim=0)
    return tensors

def process_batch(batch,vocab,device,max_relpos=20):
    inputs,lengths,masks,positions,labels=list(zip(*batch))
    astmask,sibmask,ancestormask=list(zip(*masks))
    astmask=list(astmask)
    sibmask=list(sibmask)
    ancestormask=list(ancestormask)
    childrelpos,sibrelpos,ancestorrelpos=list(zip(*positions))
    childrelpos=list(childrelpos)
    sibrelpos=list(sibrelpos)
    ancestorrelpos=list(ancestorrelpos)
    #print(len(astmask),len(childrelpos),len(inputs))
    #inputs=list(inputs)
    #for i in range(len(inputs)):
        #inputs[i]=inputs[i].to(device)
    inputbatch=pad_sequence(inputs,batch_first=True,padding_value=vocab['<pad>']).to(device)
    batch_maxlen=inputbatch.size()[1]
    #print(inputbatch.size())
    lengths=torch.tensor(lengths,dtype=torch.long).to(device)
    astmask=padmasks(astmask,batch_maxlen,device=device)
    sibmask=padmasks(sibmask,batch_maxlen,device=device)
    ancestormask=padmasks(ancestormask,batch_maxlen,device=device)
    childrelpos=padmasks(childrelpos,batch_maxlen,device=device).long()
    sibrelpos=padmasks(sibrelpos,batch_maxlen,device=device).long()
    ancestorrelpos=padmasks(ancestorrelpos,batch_maxlen,device=device).long()
    #print(astmask.size(),childrelpos.size())
    childrelpos=childrelpos.clamp(min=-max_relpos,max=max_relpos)
    #childrelpos+=max_relpos #use negative relpos
    labels=torch.tensor(labels,dtype=torch.long).to(device)
    #print(labels)
    return inputbatch,lengths,[astmask,sibmask,ancestormask],[childrelpos,sibrelpos,ancestorrelpos],labels
