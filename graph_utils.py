from collections import namedtuple, defaultdict
import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.function as fn
from copy import deepcopy
import anytree
from anytree import AnyNode, RenderTree
from preprocessc import createdata as createdata_pyc
from preprocessc import read_data_from_disk
import pickle

def tree2dgltree(tree,vocab,bidirectional=False):
    src_ids=[]
    tgt_ids=[]
    node_types=[]
    def traverse(node):
        node_types.append(vocab[node.name])
        for child in node.children:
            src_ids.append(child.index)
            tgt_ids.append(node.index)
            traverse(child)
    traverse(tree)
    node_types=torch.tensor(node_types)
    graph=dgl.graph((src_ids,tgt_ids))
    graph.ndata['type']=node_types
    if bidirectional==True:
        graph=dgl.add_reverse_edges(graph)
    return graph

def tree2dgltree_typededges(tree,vocab):
    src_ids=[]
    tgt_ids=[]
    node_types=[]
    dfs_ids=[]

    edge_types=[] #0: child->parent, 1: parent->child, 2: nextsib, 3: prevsib
    def traverse(node):
        node_types.append(vocab[node.name])
        for child in node.children:
            #g.add_nodes(1,data={'type':torch.tensor([vocab[child.name]])})
            #g.add_edges(child.index,node.index)
            src_ids.append(child.index)
            tgt_ids.append(node.index)
            edge_types.append(0)
            src_ids.append(node.index)
            tgt_ids.append(child.index)
            edge_types.append(1)
            traverse(child)
    def traverse_sib(node):
        for i in range(len(node.children)):
            if i+1<len(node.children):
                src_ids.append(node.children[i].index)
                tgt_ids.append(node.children[i+1].index)
                edge_types.append(2)
                src_ids.append(node.children[i+1].index)
                tgt_ids.append(node.children[i].index)
                edge_types.append(3)
            traverse_sib(node.children[i])
    traverse(tree)
    traverse_sib(tree)
    dfs_ids=list(range(len(node_types)))
    node_types=torch.tensor(node_types)
    dfs_ids=torch.tensor(dfs_ids)

    edge_types=torch.tensor(edge_types)
    graph=dgl.graph((src_ids,tgt_ids))
    graph.ndata['type']=node_types
    graph.ndata['dfs_id']=dfs_ids

    graph.edata['etype']=edge_types
    return graph


def tree2dgltree_loc(tree,vocab,mode='operator',bidirectional=False):
    src_ids=[]
    tgt_ids=[]
    node_types=[]
    dfs_ids=[]
    labels=[]
    is_op=[]
    def traverse(node):
        node_types.append(vocab[node.name])
        labels.append(node.label)
        is_op.append(node.is_operator)
        for child in node.children:
            src_ids.append(child.index)
            tgt_ids.append(node.index)
            traverse(child)
    traverse(tree)
    dfs_ids=list(range(len(node_types)))
    node_types=torch.tensor(node_types)
    dfs_ids=torch.tensor(dfs_ids)
    labels=torch.tensor(labels)
    is_op=torch.tensor(is_op)
    graph=dgl.graph((src_ids,tgt_ids))
    graph.ndata['type']=node_types
    graph.ndata['dfs_id']=dfs_ids
    graph.ndata['label']=labels
    graph.ndata['is_op']=is_op
    if bidirectional==True:
        graph=dgl.add_reverse_edges(graph)
    return graph

def tree2dgltree_loc_typededges(tree,vocab,mode='operator'):
    src_ids=[]
    tgt_ids=[]
    node_types=[]
    dfs_ids=[]
    labels=[]
    is_op=[]
    edge_types=[] #0: child->parent, 1: parent->child, 2: nextsib, 3: prevsib
    def traverse(node):
        node_types.append(vocab[node.name])
        labels.append(node.label)
        is_op.append(node.is_operator)
        for child in node.children:
            src_ids.append(child.index)
            tgt_ids.append(node.index)
            edge_types.append(0)
            src_ids.append(node.index)
            tgt_ids.append(child.index)
            edge_types.append(1)
            traverse(child)
    def traverse_sib(node):
        for i in range(len(node.children)):
            if i+1<len(node.children):
                src_ids.append(node.children[i].index)
                tgt_ids.append(node.children[i+1].index)
                edge_types.append(2)
                src_ids.append(node.children[i+1].index)
                tgt_ids.append(node.children[i].index)
                edge_types.append(3)
            traverse_sib(node.children[i])
    traverse(tree)
    traverse_sib(tree)
    dfs_ids=list(range(len(node_types)))
    node_types=torch.tensor(node_types)
    dfs_ids=torch.tensor(dfs_ids)
    labels=torch.tensor(labels)
    is_op=torch.tensor(is_op)
    edge_types=torch.tensor(edge_types)
    graph=dgl.graph((src_ids,tgt_ids))
    graph.ndata['type']=node_types
    graph.ndata['dfs_id']=dfs_ids
    graph.ndata['label']=labels
    graph.ndata['is_op']=is_op

    graph.edata['etype']=edge_types
    return graph

def create_graph_dataset(samples,vocab,maxlen=512,bidirectional=False,edgetypes=False):
    print('generating dataset...')
    dataset=[]
    for sample in samples:
        tree,label=sample
        if edgetypes==True:
            astgraph=tree2dgltree_typededges(tree,vocab)
        else:
            astgraph=tree2dgltree(tree,vocab,bidirectional=bidirectional)
        data=[astgraph,label]
        dataset.append(data)
    return dataset

def create_loc_dataset(samples,vocab,bidirectional=False):
    print('generating dataset...')
    dataset=[]
    for sample in samples:
        tree=sample['tree']
        astgraph=tree2dgltree_loc(tree,vocab,bidirectional=bidirectional)
        dataset.append(astgraph)
    return dataset


def create_graph_batch(batch,device):
    graphs,labels=list(zip(*batch))
    root_ids=[]
    rootid=0
    for graph in graphs:
        root_ids.append(rootid)
        rootid+=graph.number_of_nodes()
    labels=torch.tensor(labels,dtype=torch.long).to(device)
    batched_graph=dgl.batch(graphs).to(device)
    return batched_graph,labels,root_ids

def create_loc_batch(batch,device):
    graph_lengths=[]
    labels=[]
    batch_maxlen=0
    for graph in batch:
        numnodes=graph.num_nodes()
        graph_lengths.append(numnodes)
        batch_maxlen=max(batch_maxlen,numnodes)
        location=torch.nonzero(graph.ndata['label']==1,as_tuple=True)[0].item()
        labels.append(location)
    labels=torch.tensor(labels,dtype=torch.long).to(device)
    batched_graph=dgl.batch(batch).to(device)
    return batched_graph,labels,batch_maxlen,graph_lengths
