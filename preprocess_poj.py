#create poj dataset with pycparser

import os
import re
import random
import json
import h5py
import pycparser
from pycparser import c_parser
from ast2struct import ast2list
import anytree
from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from collections import deque, Counter
import pickle
from torchtext.vocab import Vocab

def get_token(node, lower=True):
        if isinstance(node, str):
            return node
        name = node.__class__.__name__
        token = name
        is_name = False
        if len(node.children()) == 0:
            attr_names = node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = node.names[0]
                elif 'name' in attr_names:
                    token = node.name
                    is_name = True
                else:
                    token = node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = node.declname
            if node.attr_names:
                attr_names = node.attr_names
                if 'op' in attr_names:
                    if node.op[0] == 'p':
                        token = node.op[1:]
                    else:
                        token = node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

def converttoany(tree,vocabcounter,keep_ids=False):
    nodeid=[0]    
    currentnodetype = tree.__class__.__name__
    token_text=get_token(tree)
    if keep_ids==False:
        root = AnyNode(name=currentnodetype,index=0, rootpath=[0])
    else:
        root = AnyNode(name=token_text,index=0, rootpath=[0])
        vocabcounter[token_text]+=1
    def traverse(tree,parent=None): #parent: current anynode for tree
        nodeid[0]+=1
        for (child_name, child) in tree.children():
            childtype = child.__class__.__name__
            token_text=get_token(child)
            if keep_ids==False:
                childnode=AnyNode(name=childtype,parent=parent,index=nodeid[0])
            else:
                childnode=AnyNode(name=token_text,parent=parent,index=nodeid[0])
                vocabcounter[token_text]+=1
            childnode.rootpath=parent.rootpath+[childnode.index] #generate path from root to current node
            traverse(child,parent=childnode)
    traverse(tree,parent=root)
    return root

def bfs(root): #write bfs order to anytree
    id=0
    q = deque()
    root.indexbfs=id
    q.append(root)
    while q:
        temp = q.popleft()
        if len(temp.children)>0:
            for child in temp.children:
                id+=1
                child.indexbfs=id
                q.append(child)

def createdata(classes=104):
    trainset=[]
    validset=[]
    testset=[]
    parser = c_parser.CParser()
    processed_data_dir='poj104_pyc/'
    train_data_path=processed_data_dir+'traindata_712.jsonl'
    dev_data_path=processed_data_dir+'devdata_712.jsonl'
    test_data_path=processed_data_dir+'testdata_712.jsonl'
    trainf=open(train_data_path, "a+")
    devf=open(dev_data_path, "a+")
    testf=open(test_data_path, "a+")
    exporter = JsonExporter(indent=2, sort_keys=True)
    wordcount=Counter()
    for i in range(1, classes+1):
        dirname = 'poj104/' + str(i) + '/' #your path of raw data
        j=0
        for rt, dirs, files in os.walk(dirname):
            for file in files:
                filename=dirname+file
                #print(filename)
                file=open(filename)
                code=file.read()
                code = code.split('\r')
                newcode = ''
                for line in code:
                    newcode = newcode + line
                code=newcode
                ast=parser.parse(code)
                #traverse and write to anytree
                anyast=converttoany(ast,vocabcounter=wordcount,keep_ids=False)
                #print(RenderTree(anyast))
                bfs(anyast)
                jsonast=exporter.export(anyast)
                sample={'tree':jsonast,'label':i-1}
                if j%10==7:
                    validset.append([anyast,i-1])
                    devf.write(json.dumps(sample)+'\n')
                elif j%10==8 or j%10==9:
                    testset.append([anyast,i-1])
                    testf.write(json.dumps(sample)+'\n')
                else:
                    trainset.append([anyast,i-1])
                    trainf.write(json.dumps(sample)+'\n')
                j+=1
                file.close()
    print(len(trainset),len(validset),len(testset))
    node_vocab=Vocab(wordcount)
    print(len(node_vocab))
    vocabfile = open('poj104_pyc/tokenvocab.pkl', 'wb')
    pickle.dump(node_vocab, vocabfile)
    vocabfile.close()
    return trainset,validset,testset

def read_data_from_disk(path):
    importer = JsonImporter()
    dataset=[]
    with open(path) as f:
        lines=f.readlines()
        for line in lines:
            data=json.loads(line)
            root = importer.import_(data['tree'])
            #print(data['label'])
            dataset.append([root,data['label']])
    return dataset
