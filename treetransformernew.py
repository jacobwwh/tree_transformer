#our tree-transformer model

from collections import namedtuple
import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.function as fn
from dgl.nn import GlobalAttentionPooling
from modules.attention import MultiHeadAttention_TUPE
from copy import deepcopy
import numpy as np


def reverse_batch(batched_graph):
    batch_numnodes=batched_graph.batch_num_nodes()
    batch_numedges=batched_graph.batch_num_edges()
    batched_graph=dgl.reverse(batched_graph)
    batched_graph.set_batch_num_nodes(batch_numnodes)
    batched_graph.set_batch_num_edges(batch_numedges)
    return batched_graph


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))+x # residual connection?


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100000):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #size: (1, max_len, d_model)
        pe.requires_grad=False
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    def position_encoding_by_index(self, x, indices):
        pe=self.pe.squeeze(0)
        pe_indices=pe[indices]
        x=x+pe_indices
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)        
        # Compute the positional encodings once in log space.
        self.pe=nn.Parameter(torch.rand([max_len,d_model]))
        self.pe.data.uniform_(-0.1, 0.1)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1),:]
        return self.dropout(x)

class TreeTransformerCell(nn.Module):
    """The tree transformer proposed by us."""
    def __init__(self, num_heads, dim_model, dim_ff=128, dropout=0.2, pos_enc='fixed'):
        super(TreeTransformerCell, self).__init__()
        self.dim_model = dim_model
        self.d_k = dim_model // num_heads
        self.h = num_heads
        self.pos_enc=pos_enc
        assert self.pos_enc in ['fixed','tupe']
        if self.pos_enc=='fixed':
            self.position_encoding=FixedPositionalEncoding(dim_model,dropout=dropout)
        
        # W_q, W_k, W_v, W_o
        self.k_linear = nn.Linear(dim_model, dim_model) #sibling attention
        self.q_linear = nn.Linear(dim_model, dim_model)
        self.v_linear = nn.Linear(dim_model, dim_model)

        self.k_linear_p = nn.Linear(dim_model, dim_model) #parental attention
        self.q_linear_p = nn.Linear(dim_model, dim_model)
        self.v_linear_p = nn.Linear(dim_model, dim_model)

        if self.pos_enc=='fixed':
            self.sib_attention=MultiHeadAttention(self.d_k,self.d_k,self.dim_model,num_heads,dropout)    
            self.parent_attention=MultiHeadAttention(self.d_k,self.d_k,self.dim_model,num_heads,dropout)
        elif self.pos_enc=='tupe':
            self.sib_attention=MultiHeadAttention_TUPE(self.d_k,self.d_k,self.dim_model,num_heads,dropout,tupe=True,pe_mode='learned')
            self.parent_attention=MultiHeadAttention_TUPE(self.d_k, self.d_k, self.dim_model, num_heads, dropout,tupe=False)
        self.ff=PositionwiseFeedForward(dim_model,dim_ff,dropout=dropout)
        self.ff.w_1.weight.data.uniform_(-0.1, 0.1)
        self.ff.w_2.weight.data.uniform_(-0.1, 0.1)
       
        #self.attn_linear=nn.Linear(self.d_k*self.h, dim_model)       
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim_model)
        
        #self.pool_linear=nn.Linear(dim_model,dim_model)

    def message_func(self, edges):
        return {'h': edges.src['h'],'children_ids':edges.src['_ID']}

    def reduce_func(self, nodes):
        x=nodes.mailbox['h']
        
        if self.pos_enc=='fixed':
            assert self.position_encoding is not None
            x=self.position_encoding(x) #sibling position encoding (do not use with tupe)
        
        q=self.q_linear(x) #sibling self_attention, can be removed
        k=self.k_linear(x)
        v=self.v_linear(x)
        residual=x       
        x,attn=self.sib_attention(q,k,v) #vanilla attention or tupe     
        x=self.layer_norm(x+residual)

        new_k=self.k_linear_p(x) #parent_children attention
        new_v=self.v_linear_p(x)
        parent_q=self.q_linear_p(nodes.data['h'].unsqueeze(1))
        residual_parent=nodes.data['h']
        parent_context, attn=self.parent_attention(parent_q,new_k,new_v) # context: [b_size x len_q(1) x dim] attn: [b_size x n_heads x len_q x len_v]
        parent_context=parent_context.squeeze(1)
        parent_context=self.layer_norm(parent_context+residual_parent)
        
        h=self.ff(parent_context)
        h=self.layer_norm(h)
        return {'h': h}

    def apply_node_func(self, nodes):
        h=nodes.data['h']      
        return {'h': h}

class TreeTransformerCell_topdown(nn.Module):
    """The tree transformer proposed by us."""
    def __init__(self, num_heads, dim_model, dim_ff=128, dropout=0.2):
        super(TreeTransformerCell_topdown, self).__init__()
        self.dim_model = dim_model
        self.d_k = dim_model // num_heads
        self.h = num_heads

        self.ff=PositionwiseFeedForward(dim_model,dim_ff,dropout=dropout)
        self.ff.w_1.weight.data.uniform_(-0.1, 0.1)
        self.ff.w_2.weight.data.uniform_(-0.1, 0.1)

        #self.attn_linear=nn.Linear(self.d_k*self.h, dim_model)       
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim_model)
        
        #self.pool_linear=nn.Linear(dim_model,dim_model)

    def message_func(self, edges):
        return {'h': edges.src['h'],'children_ids':edges.src['_ID']}
    
    def reduce_topdown(self,nodes):
        parent_td_states=nodes.mailbox['h']
        children_bu_states=nodes.data['h']
        children_context=self.layer_norm(parent_td_states.squeeze()+children_bu_states) #add residual (children_bu_states)
        
        h=self.ff(children_context)
        h=self.layer_norm(h)

        return {'h': h}

    def apply_node_func(self, nodes):
        h=nodes.data['h']      
        return {'h': h}

class TreeTransformerClassifier(torch.nn.Module):
    def __init__(self, num_heads, dim_model, dim_hidden, n_classes, vocab_size, dropout=0.2, num_stacks=1):
        super(TreeTransformerClassifier, self).__init__()
        self.num_heads=num_heads
        self.dim_model=dim_model
        self.dropout = torch.nn.Dropout(dropout)
        self.position_encoding=LearnedPositionalEncoding(dim_model,dropout=dropout)
        self.cell=TreeTransformerCell(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout,learned_posenc=self.position_encoding)
        self.cell_topdown=TreeTransformerCell_topdown(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout,learned_posenc=self.position_encoding)
        self.embeddings=nn.Embedding(vocab_size,dim_model)
        self.classifier=nn.Linear(dim_model,n_classes)
        self.num_stacks = num_stacks
        self.pooling=GlobalAttentionPooling(nn.Linear(dim_model,1))
        self.top_down=True
        print('top_down',self.top_down)
        self.pool_mode='attention'
        print(self.pool_mode)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.cell.k_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.q_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.v_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.k_linear_p.weight.data.uniform_(-initrange, initrange)
        self.cell.q_linear_p.weight.data.uniform_(-initrange, initrange)
        self.cell.v_linear_p.weight.data.uniform_(-initrange, initrange)
        #self.cell.ff.w_1.weight.data.uniform_(-initrange, initrange)
        #self.cell.ff.w_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch):
        batch.ndata['h']=self.embeddings(batch.ndata['type'])

        for i in range(self.num_stacks):
            dgl.prop_nodes_topo(batch,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            if self.top_down:
                batch=reverse_batch(batch) #convert input trees to top-down
                dgl.prop_nodes_topo(batch,
                                message_func=self.cell_topdown.message_func,
                                reduce_func=self.cell_topdown.reduce_topdown,
                                apply_node_func=self.cell_topdown.apply_node_func)
                batch=reverse_batch(batch)
       
        batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_pred=self.classifier(batch_pred)
        return batch_pred


class TreeTransformerEncoder(torch.nn.Module):
    def __init__(self, num_heads, dim_model, dim_hidden, vocab_size, dropout=0.2):
        super(TreeTransformerEncoder, self).__init__()
        self.num_heads=num_heads
        self.dim_model=dim_model
        self.dropout = torch.nn.Dropout(dropout)
        self.position_encoding=LearnedPositionalEncoding(dim_model,dropout=dropout)
        self.cell=TreeTransformerCell(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout,learned_posenc=self.position_encoding)
        self.cell_topdown=TreeTransformerCell_topdown(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout,learned_posenc=self.position_encoding)
        self.embeddings=nn.Embedding(vocab_size,dim_model)
        self.pooling=GlobalAttentionPooling(nn.Linear(dim_model,1))
        self.top_down=True
        print('top_down',self.top_down)
        self.pool_mode='attention'
        print(self.pool_mode)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.cell.k_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.q_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.v_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.k_linear_p.weight.data.uniform_(-initrange, initrange)
        self.cell.q_linear_p.weight.data.uniform_(-initrange, initrange)
        self.cell.v_linear_p.weight.data.uniform_(-initrange, initrange)
        #self.cell.ff.w_1.weight.data.uniform_(-initrange, initrange)
        #self.cell.ff.w_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch,root_ids=None):
        batch.ndata['h']=self.embeddings(batch.ndata['type'])
        dgl.prop_nodes_topo(batch,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        if self.top_down:
            batch=reverse_batch(batch) #convert input trees to top-down
            dgl.prop_nodes_topo(batch,
                            message_func=self.cell_topdown.message_func,
                            reduce_func=self.cell_topdown.reduce_topdown,
                            apply_node_func=self.cell_topdown.apply_node_func)
            batch=reverse_batch(batch)
        return batch

class TreeTransformer_localize(torch.nn.Module):
    def __init__(self, num_heads, dim_model, dim_hidden, vocab_size, dropout=0.2):
        super(TreeTransformer_localize, self).__init__()
        self.encoder=TreeTransformerEncoder(num_heads, dim_model, dim_hidden, vocab_size, dropout)
        self.cls = nn.Linear(dim_model, 1)

    def forward(self, batch,batch_maxlen,batch_numnodes,repair=False):
        batch=self.encoder(batch)
        batch.ndata['prob']=self.cls(batch.ndata['h']).squeeze(1)
        batch.ndata['prob'] = batch.ndata['prob'].masked_fill(batch.ndata['is_op'] == False, -1e9) #mask non-operator
        
        probs=batch.ndata['prob']
        graph_probs=list(probs.split(batch_numnodes))
        graph_probs=pad_sequence(graph_probs,batch_first=True,padding_value=-1e9)
        graph_probs=F.log_softmax(graph_probs,dim=1)
        
        if repair==False:
            return graph_probs
        else:
            buggy_locations=torch.nonzero(batch.ndata['label']==1,as_tuple=True)
            repair_pred=batch.ndata['h'][buggy_locations]
            repair_pred=self.repair_cls(repair_pred)
            return graph_probs,repair_pred


class TreeTransformer_typeandtoken(torch.nn.Module):
    def __init__(self, num_heads, dim_model, d_k, d_v, dim_hidden, n_classes, token_vocabsize, type_vocabsize, dropout=0.2, num_stacks=1,top_down=True):
        super(TreeTransformer_typeandtoken, self).__init__()
        self.num_heads=num_heads
        self.dim_model=dim_model
        self.dropout = torch.nn.Dropout(dropout)
        self.position_encoding=LearnedPositionalEncoding(dim_model,dropout=dropout)
        self.cell=TreeTransformerCell(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout,learned_posenc=self.position_encoding)
        self.cell_topdown=TreeTransformerCell_topdown(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout,learned_posenc=self.position_encoding)
        self.token_embeddings=nn.Embedding(token_vocabsize,dim_model//2)
        self.type_embeddings=nn.Embedding(type_vocabsize,dim_model//2)
        print(self.token_embeddings)
        print(self.type_embeddings)
        self.classifier=nn.Linear(dim_model,n_classes)
        self.num_stacks = num_stacks
        self.pooling=GlobalAttentionPooling(nn.Linear(dim_model,1))
        self.top_down=top_down
        print('top_down',self.top_down)
        self.pool_mode='attention'
        print(self.pool_mode)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.type_embeddings.weight.data.uniform_(-initrange, initrange)
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.cell.k_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.q_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.v_linear.weight.data.uniform_(-initrange, initrange)
        self.cell.k_linear_p.weight.data.uniform_(-initrange, initrange)
        self.cell.q_linear_p.weight.data.uniform_(-initrange, initrange)
        self.cell.v_linear_p.weight.data.uniform_(-initrange, initrange)
        #self.cell.ff.w_1.weight.data.uniform_(-initrange, initrange)
        #self.cell.ff.w_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch):
        batch.ndata['h']=torch.cat([self.type_embeddings(batch.ndata['type']),self.token_embeddings(batch.ndata['token'])],dim=1)

        for i in range(self.num_stacks):
            dgl.prop_nodes_topo(batch,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            if self.top_down:
                batch=reverse_batch(batch) #convert input trees to top-down
                dgl.prop_nodes_topo(batch,
                                message_func=self.cell_topdown.message_func,
                                reduce_func=self.cell_topdown.reduce_topdown,
                                apply_node_func=self.cell_topdown.apply_node_func)
                batch=reverse_batch(batch)
        batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_pred=self.classifier(batch_pred)
        return batch_pred
