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

def clones(module, k):
    return nn.ModuleList(
        deepcopy(module) for _ in range(k)
    )

class Tree:
    def __init__(self, h_size):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size

    def add_node(self, parent_id=None, tensor:torch.Tensor = torch.Tensor()):
        self.dgl_graph.add_nodes(1, data={'x': tensor.unsqueeze(0),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        if parent_id is not None:
            self.dgl_graph.add_edge(added_node_id, parent_id)
        return added_node_id

    def add_node_bottom_up(self, child_ids, tensor: torch.Tensor):
        self.dgl_graph.add_nodes(1, data={'x': tensor.unsqueeze(0),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        for child_id in child_ids:
            self.dgl_graph.add_edge(child_id, added_node_id)
        return added_node_id

    def add_link(self, child_id, parent_id):
        self.dgl_graph.add_edge(child_id, parent_id)


class BatchedTree:
    def __init__(self, tree_list):
        graph_list = []
        for tree in tree_list:
            graph_list.append(tree.dgl_graph)
        self.batch_dgl_graph = dgl.batch(graph_list)

    def get_hidden_state(self):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        hidden_states = []
        for graph in graph_list:
            print(graph.nodes())
        max_nodes_num = max([len(graph.nodes()) for graph in graph_list])
        for graph in graph_list:
            hiddens = graph.ndata['h']
            node_num, hidden_num = hiddens.size()
            if len(hiddens) < max_nodes_num:
                padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                hiddens = torch.cat((hiddens, padding), dim=0)
            hidden_states.append(hiddens)
        return torch.stack(hidden_states)

def tree2dgltree(tree,vocab,bidirectional=False):
    src_ids=[]
    tgt_ids=[]
    node_types=[]
    dfs_ids=[]
    sib_ids=[]
    dfs_ids=[]
    def traverse(node):
        node_types.append(vocab[node.name])
        #dfs_ids.append(node.index)
        for child in node.children:
            src_ids.append(child.index)
            tgt_ids.append(node.index)
            traverse(child)
    traverse(tree)
    dfs_ids=list(range(len(node_types)))
    node_types=torch.tensor(node_types)
    dfs_ids=torch.tensor(dfs_ids)
    graph=dgl.graph((src_ids,tgt_ids))
    graph.ndata['type']=node_types
    graph.ndata['dfs_id']=dfs_ids
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

def create_cs_dataset(samples,vocab,maxlen=512): #not used
    print('generating dataset...')
    dataset=[]
    
    for sample in samples:
        if len(sample)==4:
            idx,url,tree,nl=sample
        else:
            tree,nl=sample
        nl=nl[:maxlen]
        astgraph=tree2dgltree(tree,vocab)

        if len(sample)==4:
            data=[idx,url,astgraph,nl]
        else:
            data=[astgraph,nl]
        dataset.append(data)
    return dataset

def create_cs_dataset_seq(samples,vocab,maxlen=512):
    print('generating dataset...')
    dataset=[]   
    for sample in samples:
        if len(sample)==4:
            idx,url,code,nl=sample
        else:
            code,nl=sample
        code=code[:maxlen]
        nl=nl[:maxlen]
        if len(sample)==4:
            data=[idx,url,code,nl]
        else:
            data=[code,nl]
        dataset.append(data)
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

def create_cs_batch(batch,vocab,device): #not used
    if len(batch[0])==4:
        idxs,urls,graphs,queries=list(zip(*batch))
    else:
        graphs,queries=list(zip(*batch))
    root_ids=[]
    rootid=0
    for graph in graphs:
        root_ids.append(rootid)
        rootid+=graph.number_of_nodes()
    #labels=torch.tensor(labels,dtype=torch.long).to(device)
    batched_graph=dgl.batch(graphs).to(device)

    nlbatch=[]
    batch_maxlen=0
    data_lengths=[]
    for data in queries:
        if len(data)>batch_maxlen:
            batch_maxlen= len(data)
        data_lengths.append(len(data))
        inputtensor=torch.tensor([vocab[token] for token in data],dtype=torch.long).to(device)
        nlbatch.append(inputtensor)
    nlbatch=pad_sequence(nlbatch,batch_first=True,padding_value=vocab['<pad>'])
    padding_mask=torch.zeros(len(queries),batch_maxlen).to(device)
    for i in range(len(queries)):
        padding_mask[i, data_lengths[i]:] = 1
    padding_mask=padding_mask.bool()
    return batched_graph,root_ids,nlbatch,padding_mask

def create_cs_batch_seq(batch,vocab,device): #not used
    if len(batch[0])==4:
        idxs,urls,programs,queries=list(zip(*batch))
    else:
        programs,queries=list(zip(*batch))
    codebatch=[]
    codebatch_maxlen=0
    code_lengths=[]
    for data in programs:
        if len(data)>codebatch_maxlen:
            codebatch_maxlen= len(data)
        code_lengths.append(len(data))
        inputtensor=torch.tensor([vocab[token] for token in data],dtype=torch.long).to(device)
        codebatch.append(inputtensor)
    codebatch=pad_sequence(codebatch,batch_first=True,padding_value=vocab['<pad>'])
    code_padding_mask=torch.zeros(len(programs),codebatch_maxlen).to(device)
    for i in range(len(queries)):
        code_padding_mask[i, code_lengths[i]:] = 1
    code_padding_mask=code_padding_mask.bool()

    nlbatch=[]
    batch_maxlen=0
    data_lengths=[]
    for data in queries:
        if len(data)>batch_maxlen:
            batch_maxlen= len(data)
        data_lengths.append(len(data))
        inputtensor=torch.tensor([vocab[token] for token in data],dtype=torch.long).to(device)
        nlbatch.append(inputtensor)
    nlbatch=pad_sequence(nlbatch,batch_first=True,padding_value=vocab['<pad>'])
    nl_padding_mask=torch.zeros(len(queries),batch_maxlen).to(device)
    for i in range(len(queries)):
        nl_padding_mask[i, data_lengths[i]:] = 1
    nl_padding_mask=nl_padding_mask.bool()
    return codebatch,code_padding_mask,nlbatch,nl_padding_mask

def create_graph_batch_sst(batch,device): #not used
    batch_rootids=[]
    graphs=[]
    cur_rootid=0
    for graph, rootid in batch:
        batch_rootids.append(cur_rootid+rootid)
        cur_rootid+=graph.number_of_nodes()
        graphs.append(graph)
    batched_graph=dgl.batch(graphs).to(device)
    node_labels=batched_graph.ndata['label']
    root_labels=batched_graph.ndata['label'][batch_rootids]   

    node_labeledids=(node_labels!=-3).nonzero(as_tuple=False).squeeze(1) #indices of nodes with sentiment labels
    node_labels=node_labels[node_labeledids]
    batch_rootids=torch.tensor(batch_rootids,dtype=torch.long).to(device)
    return batched_graph, node_labeledids, node_labels, batch_rootids, root_labels

class sstdataiterator(object):
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
