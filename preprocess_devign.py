# create devign dataset with tree-sitter

import os
import re
import random
import json
import pycparser
from pycparser import c_parser
from tree_sitter import Language, Parser
from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from collections import defaultdict,deque,Counter
import pickle
from torchtext.vocab import Vocab

C_LANGUAGE = Language('/var/data/wangwh/sitter/build/my-languages.so', 'c')
sitter_parser = Parser()
sitter_parser.set_language(C_LANGUAGE)

def removeComments(source):#source: code separated by lines
    res = []
    multi = False
    line = ''
    for s in source:
        i = 0
        while i < len(s):
            if not multi:
                if s[i] == '/' and i < len(s) - 1 and s[i + 1] == '/':
                    break
                elif s[i] == '/' and i < len(s) - 1 and s[i + 1] == '*':
                    multi = True
                    i += 1
                else:
                    line += s[i]
            else:
                if s[i] == '*' and i < len(s) - 1 and s[i + 1] == '/':
                    multi = False
                    i += 1
            i += 1
        if not multi and line:
            res.append(line)
            line = ''
    res='\n'.join(res)
    return res

def tokenize_with_camel_case(token):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]

def tokenize_with_snake_case(token):
    return token.split('_')

def tokenize_camelsnake(token):
    tokenized=[]
    snake_tokens=tokenize_with_snake_case(token)
    for token in snake_tokens:
        tokenized.extend(tokenize_with_camel_case(token))
    return tokenized

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

terminaltypes=['identifier','number_literal','primitive_type','type_identifier','char_literal','string_literal']
operators=['+','-','*','/','++','--','%','||','&&','|','^','&','==','!=','>','>=','<=','<','<<','>>']
vocab_count=defaultdict(int)

def converttoany(tree,text,vocabcounter,abstract=True,build_vocab=True,camelsnake=False):
    nodeid=[0]
    currentnodetype = tree.type
    root = AnyNode(name=currentnodetype,index=0)
    vocab_count[currentnodetype]+=1
    def traverse(tree,parent=None): #parent: current anynode for tree
        nodeid[0]+=1
        for child in tree.children:
            childtype = child.type
            if abstract==False:
                if childtype in terminaltypes:
                    nodename=text[child.start_byte:child.end_byte].decode('utf-8')
                else:
                    nodename=childtype
                vocab_count[nodename]+=1
                vocabcounter[nodename]+=1
                childnode=AnyNode(name=nodename,parent=parent,index=nodeid[0])
                traverse(child,parent=childnode)
            else:
                if child.is_named or child.type in operators:
                    if childtype in terminaltypes:
                        nodename=text[child.start_byte:child.end_byte].decode('utf-8')
                    else:
                        nodename=childtype
                    vocab_count[nodename]+=1
                    vocabcounter[nodename]+=1
                    childnode=AnyNode(name=nodename,parent=parent,index=nodeid[0])
                    if camelsnake==True:
                        if len(child.children)==0 and child.type=='identifier':#child is leaf/identifier
                            id_subtokens=tokenize_camelsnake(nodename)
                            if len(id_subtokens)>1:
                                for subtoken in id_subtokens:
                                    nodeid[0]+=1
                                    subnode=AnyNode(name=subtoken,parent=childnode,index=nodeid[0])
                    traverse(child,parent=childnode)
    traverse(tree,parent=root)
    return root

def createdata_devign(vocab_size=None):
    trainset=[]
    validset=[]
    testset=[]
    trainfile='/var/data/wangwh/CodeXGLUE/Code-Code/Defect-detection/dataset/train.jsonl'
    validfile='/var/data/wangwh/CodeXGLUE/Code-Code/Defect-detection/dataset/valid.jsonl'
    testfile='/var/data/wangwh/CodeXGLUE/Code-Code/Defect-detection/dataset/test.jsonl'
    parser = c_parser.CParser()
    processed_data_dir='devign/'
    train_data_path=processed_data_dir+'traindata_camelsnake.jsonl'
    dev_data_path=processed_data_dir+'devdata_camelsnake.jsonl'
    test_data_path=processed_data_dir+'testdata_camelsnake.jsonl'
    exporter = JsonExporter(indent=2, sort_keys=True)
    trainf=open(train_data_path, "a+")
    devf=open(dev_data_path, "a+")
    testf=open(test_data_path, "a+")
    wordcount=Counter()

    with open(trainfile) as f:
        i=0
        lines=f.readlines()
        for line in lines:
            jline=json.loads(line)
            label=jline['target']
            sourcecode=jline['func']
            sourcecode = sourcecode.split('\r')
            newcode = ''
            for line in sourcecode:
                newcode = newcode + line
            sourcecode=newcode
            codelines=sourcecode.split('\n')
            numlines=len(codelines)
            if numlines<10000:
                sourcecode=removeComments(codelines) #remove comments
                bytescode=bytes(sourcecode,'utf-8')
                sitter_ast=sitter_parser.parse(bytescode)
                anyast=converttoany(sitter_ast.root_node,bytescode,vocabcounter=wordcount,abstract=True,camelsnake=True)
                trainset.append([anyast,label])
                jsonast=exporter.export(anyast)
                sample={'tree':jsonast,'label':label}
                trainf.write(json.dumps(sample)+'\n')
    with open(validfile) as f:
        lines=f.readlines()
        for line in lines:
            jline=json.loads(line)
            label=jline['target']
            sourcecode=jline['func']
            sourcecode = sourcecode.split('\r')
            newcode = ''
            for line in sourcecode:
                newcode = newcode + line
            sourcecode=newcode
            codelines=sourcecode.split('\n')
            numlines=len(codelines)
            if numlines<10000:
                sourcecode=removeComments(codelines)
                bytescode=bytes(sourcecode,'utf-8')
                sitter_ast=sitter_parser.parse(bytescode)
                anyast=converttoany(sitter_ast.root_node,bytescode,vocabcounter=wordcount,abstract=True,camelsnake=True)
                validset.append([anyast,label])
                jsonast=exporter.export(anyast)
                sample={'tree':jsonast,'label':label}
                devf.write(json.dumps(sample)+'\n')
    with open(testfile) as f:
        lines=f.readlines()
        for line in lines:
            jline=json.loads(line)
            label=jline['target']
            sourcecode=jline['func']
            sourcecode = sourcecode.split('\r')
            newcode = ''
            for line in sourcecode:
                newcode = newcode + line
            sourcecode=newcode
            codelines=sourcecode.split('\n')
            numlines=len(codelines)
            if numlines<10000:
                sourcecode=removeComments(codelines)
                bytescode=bytes(sourcecode,'utf-8')
                sitter_ast=sitter_parser.parse(bytescode)
                anyast=converttoany(sitter_ast.root_node,bytescode,vocabcounter=wordcount,abstract=True,camelsnake=True)
                testset.append([anyast,label])
                jsonast=exporter.export(anyast)
                sample={'tree':jsonast,'label':label}
                testf.write(json.dumps(sample)+'\n')
    print(len(trainset),len(validset),len(testset))
    #vocablen=len(vocab_count)
    #print(vocablen)
    #indices=list(range(vocablen+1)) #save vocab
    '''node_vocab=Vocab(wordcount)
    print(len(node_vocab))
    vocabfile = open('devign/devignvocab_cst.pkl', 'wb')
    pickle.dump(node_vocab, vocabfile)
    vocabfile.close()'''
    node_vocab=Vocab(wordcount,min_freq=5)
    print(len(node_vocab))
    vocabfile = open('devign/devignvocab_ast_camelsnake_min5.pkl', 'wb')
    pickle.dump(node_vocab, vocabfile)
    vocabfile.close()
    node_vocab=Vocab(wordcount,min_freq=2)
    print(len(node_vocab))
    vocabfile = open('devign/devignvocab_ast_camelsnake_min2.pkl', 'wb')
    pickle.dump(node_vocab, vocabfile)
    vocabfile.close()
    '''node_vocab=Vocab(wordcount,max_size=20000)
    print(len(node_vocab))
    vocabfile = open('devign/devignvocab_ast_min2.pkl', 'wb')
    pickle.dump(node_vocab, vocabfile)
    vocabfile.close()'''
    #sorted(vocab_count.items(), key=lambda d:d[1], reverse=True)
    '''words=list(vocab_count.keys())+['<UNK>']
    vocab_final=dict(zip(words,indices))
    vocabfile = open('devign/devignvocab_ast.pkl', 'wb')
    pickle.dump(vocab_final, vocabfile)
    vocabfile.close()'''
    return trainset,validset,testset


if __name__ == '__main__':
    createdata_devign()
