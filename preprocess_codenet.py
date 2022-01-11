#preprocess codenet data

import csv
import torch
import dgl
import dgl.function as fn
from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from collections import defaultdict,deque,Counter
import pickle
from torchtext.vocab import Vocab

pythonpath='Project_CodeNet/derived/benchmarks/Project_CodeNet_Python800_spts/' #change to your own data path
c1000path='Project_CodeNet/derived/benchmarks/Project_CodeNet_C++1000_spts/'
c1400path='Project_CodeNet/Project_CodeNet/derived/benchmarks/Project_CodeNet_C++1400_spts/'
javapath='Project_CodeNet/Project_CodeNet/derived/benchmarks/Project_CodeNet_Java250_spts/'


def get_spt_dataset(bidirection=False, virtual=False,edgetype=False,next_token=False,data='java250'):
    assert data in ['java250','c++1000','c++1400','python800']
    if data=='java250':
        datapath=javapath
    elif data=='c++1000':
        datapath=c1000path
    elif data=='c++1400':
        datapath=c1400path
    elif data=='python800':
        datapath=pythonpath
    edgepath=datapath+'edge.csv'
    labelpath=datapath+'graph-label.csv'
    nodepath=datapath+'node-feat.csv'
    edgenumpath=datapath+'num-edge-list.csv'
    nodenumpath=datapath+'num-node-list.csv'
    
    numnodes=[]
    numedges=[]
    nodefeats=[]
    is_tokens=[]
    token_ids=[]
    rule_ids=[]
    edges=[]
    labels=[]
    token_vocabsize=0
    type_vocabsize=0
    with open(nodenumpath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            numnodes.append(int(row[0]))
    with open(edgenumpath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            numedges.append(int(row[0]))
    with open(labelpath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            labels.append(int(row[0]))
    print(len(numnodes),len(numedges),len(labels))
    with open(edgepath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            source,target=row
            source,target=int(source),int(target)
            edges.append([source,target])
    print(len(edges))
    with open(nodepath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            is_token,token_type,rule_type,is_reserved=row
            token_type,rule_type=int(token_type),int(rule_type)
            if token_type>token_vocabsize:
                token_vocabsize=token_type
            if rule_type>type_vocabsize:
                type_vocabsize=rule_type
            nodefeats.append([token_type,rule_type])
            token_ids.append(token_type)
            rule_ids.append(rule_type)
            is_tokens.append(int(is_token))
    print(len(nodefeats))
    
    all_graphdata=[]
    graph_nodestart=0
    graph_edgestart=0
    if next_token==True:
        assert edgetype==True
    if edgetype==True:
        for i in range(len(labels)):
            num_node,num_edge,graph_label=numnodes[i],numedges[i],labels[i]
            graph_edge=edges[graph_edgestart:graph_edgestart+num_edge]
            
            graph_istoken=is_tokens[graph_nodestart:graph_nodestart+num_node]
            token_ids=[i for i in range(num_node) if graph_istoken[i]==1]
            
            targets,sources=list(zip(*graph_edge)) #from child to parent
            targets,sources=list(targets),list(sources)
            edge_types=[0]*len(targets)+[1]*len(targets) #0: child->parent, 1: parent->child, 2: nextsib, 3: prevsib
            targets,sources=targets+sources,sources+targets #add parent->child edges
            for i in range(len(graph_edge)): #add sibling edges
                parentid,childid=graph_edge[i]
                if i>0 and parentid==graph_edge[i-1][0]: #have same parent
                    sources.append(graph_edge[i-1][1])
                    targets.append(graph_edge[i][1])
                    edge_types.append(2)
                    sources.append(graph_edge[i][1])
                    targets.append(graph_edge[i-1][1])
                    edge_types.append(3)
            g=dgl.graph((sources,targets))
            graph_tokens=torch.tensor(token_ids[graph_nodestart:graph_nodestart+num_node])
            graph_rules=torch.tensor(rule_ids[graph_nodestart:graph_nodestart+num_node])
            edge_types=torch.tensor(edge_types)
            g.ndata['token']=graph_tokens
            g.ndata['type']=graph_rules
            g.edata['etype']=edge_types
            graph_nodestart+=num_node
            graph_edgestart+=num_edge
            all_graphdata.append([g,graph_label])
    else:
        for i in range(len(labels)):
            num_node,num_edge,graph_label=numnodes[i],numedges[i],labels[i]
            graph_edge=edges[graph_edgestart:graph_edgestart+num_edge]
            targets,sources=list(zip(*graph_edge)) #from child to parent
            if bidirection==True: # bidirectional graph for gnn
                targets,sources=targets+sources,sources+targets
            targets,sources=torch.tensor(targets),torch.tensor(sources)
            g=dgl.graph((sources,targets))
            graph_tokens=torch.tensor(token_ids[graph_nodestart:graph_nodestart+num_node])
            graph_rules=torch.tensor(rule_ids[graph_nodestart:graph_nodestart+num_node])
            g.ndata['token']=graph_tokens
            g.ndata['type']=graph_rules
            graph_nodestart+=num_node
            graph_edgestart+=num_edge
            all_graphdata.append([g,graph_label])

    #simple data split
    print(len(all_graphdata))
    trainset,devset,testset=[[],[],[]]
    for i in range(len(all_graphdata)):
        if i%5==3:
            devset.append(all_graphdata[i])
        elif i%5==4:
            testset.append(all_graphdata[i])
        else:
            trainset.append(all_graphdata[i])
    print(len(trainset),len(devset),len(testset))
    token_vocabsize+=2
    type_vocabsize+=2
    print(token_vocabsize,type_vocabsize)
    return trainset,devset,testset,token_vocabsize,type_vocabsize
