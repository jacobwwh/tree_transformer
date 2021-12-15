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
from propagate import prop_nodes_topdown
#from classifier import BasicTransformer, BasicTransformerEncoder
from copy import deepcopy
import numpy as np

import copy
def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1, top_down=False):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.top_down=top_down
        #print('attetnion topdown', self.top_down)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        #print(q.size(),k.size(),v.size())
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        #print(scores.size())
        #print(scores)
        #print(self.softmax(scores))
        if self.top_down==True:
            #newsoftmax = nn.Softmax(dim=-2)
            #print(scores)
            #print(self.softmax(scores))
            #attn=self.dropout(F.sigmoid(scores))
            attn = self.dropout(self.softmax(scores))
            #print(attn.size(),v.size())
            #print(attn)
            #quit()
            #attn=self.dropout(scores)           
            context = torch.matmul(attn, v)
        else:
            attn = self.dropout(self.softmax(scores))
            context = torch.matmul(attn, v)
        #print(attn)
        # outputs: [b_size x n_heads x len_q x d_v]
        #print(context)
        #quit()
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, top_down=False):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention = ScaledDotProductAttention(d_k, dropout,top_down=top_down)
        self.top_down=top_down

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = q.view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = k.view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = v.view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn

class MultiBranchAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_branches, dropout):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_branches = n_branches

        self.multihead_attn = MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        # additional parameters for BranchedAttention
        self.w_o = nn.ModuleList([nn.Linear(d_v, d_model) for _ in range(n_branches)])
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp/self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a/self.w_a.sum())

        #self.pos_ffn = nn.ModuleList([PoswiseFeedForwardNet(d_model, d_ff//n_branches, dropout) for _ in range(n_branches)])
        #self.dropout = nn.Dropout(dropout)
        #self.layer_norm = LayerNormalization(d_model)
        #nn.init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask=None):
        # context: a tensor of shape [b_size x len_q x n_branches * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # context: a list of tensors of shape [b_size x len_q x d_v] len: n_branches
        context = context.split(self.d_v, dim=-1)

        # outputs: a list of tensors of shape [b_size x len_q x d_model] len: n_branches
        outputs = [self.w_o[i](context[i]) for i in range(self.n_branches)]
        outputs = [kappa * output for kappa, output in zip(self.w_kp, outputs)]
        return outputs, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))+x # residual connection?

class PoswiseFF_Conv(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFF_Conv, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

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
    def __init__(self, num_heads, dim_model, dim_ff=128, dropout=0.2,attn_type='multihead',learned_posenc=None):
        super(TreeTransformerCell, self).__init__()
        self.dim_model = dim_model
        self.d_k = dim_model // num_heads
        self.h = num_heads
        self.pos_enc='fixed'
        print('position encoding:',self.pos_enc)
        assert self.pos_enc in ['fixed','learned']
        if self.pos_enc=='fixed':
            self.position_encoding=FixedPositionalEncoding(dim_model,dropout=dropout)
        elif self.pos_enc=='learned':               
            self.position_encoding=learned_posenc
        #self.position_encoding=LearnedPositionalEncoding(dim_model,dropout=dropout)
        print('fixed position encoding: sibling positions')
        self.bottom_up_pos=False
        print('bottom up position',self.bottom_up_pos)
        # W_q, W_k, W_v, W_o
        self.k_linear = nn.Linear(dim_model, dim_model) #sibling attention, not used
        self.q_linear = nn.Linear(dim_model, dim_model)
        self.v_linear = nn.Linear(dim_model, dim_model)

        self.k_linear_p = nn.Linear(dim_model, dim_model)
        self.q_linear_p = nn.Linear(dim_model, dim_model)
        self.v_linear_p = nn.Linear(dim_model, dim_model)


        self.sib_attention=MultiHeadAttention(self.d_k,self.d_k,self.dim_model,num_heads,dropout)
        self.parent_attention=MultiHeadAttention(self.d_k,self.d_k,self.dim_model,num_heads,dropout)
        self.ff=PositionwiseFeedForward(dim_model,dim_ff,dropout=dropout)
        self.ff.w_1.weight.data.uniform_(-0.1, 0.1)
        self.ff.w_2.weight.data.uniform_(-0.1, 0.1)
       
        self.attn_linear=nn.Linear(self.d_k*self.h, dim_model)       
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim_model)
        
        self.pool_linear=nn.Linear(dim_model,dim_model)

    def message_func(self, edges):
        return {'h': edges.src['h'],'children_ids':edges.src['_ID']}

    def reduce_func(self, nodes):
        x=nodes.mailbox['h']
        assert self.position_encoding is not None
        if self.bottom_up_pos:
            x=self.position_encoding(x) #sibling position

        new_k=self.k_linear_p(x) #parent_children attention
        new_v=self.v_linear_p(x)
        parent_q=self.q_linear_p(nodes.data['h'].unsqueeze(1))

        residual_parent=parent_q.squeeze(1)

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
    def __init__(self, num_heads, dim_model, dim_ff=128, dropout=0.2,attn_type='multihead',learned_posenc=None):
        super(TreeTransformerCell_topdown, self).__init__()
        self.dim_model = dim_model
        self.d_k = dim_model // num_heads
        self.h = num_heads
        self.pos_enc='fixed'
        print('position encoding:',self.pos_enc)
        assert self.pos_enc in ['fixed','learned']
        if self.pos_enc=='fixed':
            self.position_encoding=FixedPositionalEncoding(dim_model,dropout=dropout)
        elif self.pos_enc=='learned':               
            self.position_encoding=learned_posenc
        self.top_down_pos=False
        print('top down position',self.top_down_pos)
        # W_q, W_k, W_v, W_o
        self.k_linear = nn.Linear(dim_model, dim_model)
        self.q_linear = nn.Linear(dim_model, dim_model)
        self.v_linear = nn.Linear(dim_model, dim_model)

        self.k_linear_p = nn.Linear(dim_model, dim_model)
        self.q_linear_p = nn.Linear(dim_model, dim_model)
        self.v_linear_p = nn.Linear(dim_model, dim_model)

        self.sib_attention=MultiHeadAttention(self.d_k,self.d_k,self.dim_model,num_heads,dropout,top_down=True)
        self.parent_attention=MultiHeadAttention(self.d_k,self.d_k,self.dim_model,num_heads,dropout,top_down=True)
        self.ff=PositionwiseFeedForward(dim_model,dim_ff,dropout=dropout)
        self.ff.w_1.weight.data.uniform_(-0.1, 0.1)
        self.ff.w_2.weight.data.uniform_(-0.1, 0.1)

        self.attn_linear=nn.Linear(self.d_k*self.h, dim_model)       
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim_model)
        
        self.pool_linear=nn.Linear(dim_model,dim_model)

    def message_func(self, edges):
        return {'h': edges.src['h'],'children_ids':edges.src['_ID']}

    def reduce_func(self, graph, nodes):
        x=nodes.mailbox['h']
        if self.top_down_pos:
            x=self.position_encoding(x) #sibling position encoding

        #assert self.position_encoding is not None #position encoding before attention
        #x=self.position_encoding(x)

        new_k=self.k_linear_p(nodes.data['h'].unsqueeze(1)) #parent_children attention
        new_v=self.v_linear_p(nodes.data['h'].unsqueeze(1))
        children_q=self.q_linear_p(x)

        residual_children=children_q

        children_context, attn=self.parent_attention(children_q,new_k,new_v) # context: [b_size x len_q(1) x dim] attn: [b_size x n_heads x len_q x len_v]
        #print(children_context.size(),residual_children.size())
        children_context=self.layer_norm(children_context+residual_children)
        h=self.ff(children_context)
        h=self.layer_norm(h)

        graph.ndata['h'][nodes.mailbox['children_ids']]=h
        return {'h': nodes.data['h']}

    def apply_node_func(self, nodes):
        h=nodes.data['h']      
        return {'h': h}

class TreeTransformerClassifier(torch.nn.Module):
    def __init__(self, num_heads, dim_model, d_k, d_v, dim_hidden, n_classes, vocab_size, dropout=0.2, num_stacks=1):
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

    def forward(self, batch,root_ids=None):
        batch.ndata['h']=self.embeddings(batch.ndata['type'])
        #batch.ndata['h']=self.cell.position_encoding.position_encoding_by_index(batch.ndata['h'],dfs_positions) #dfs position encoding (do not use)

        for i in range(self.num_stacks):
            dgl.prop_nodes_topo(batch,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            if self.top_down:
                prop_nodes_topdown(batch,
                                    message_func=self.cell_topdown.message_func,
                                    reduce_func=self.cell_topdown.reduce_func,
                                    apply_node_func=self.cell_topdown.apply_node_func)

        if self.pool_mode=='root':
            batch_pred=batch.ndata['h'][root_ids]
        else:
            batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_pred=self.classifier(batch_pred)
        return batch_pred
    
    def forward_nodeclassification(self, batch, node_ids, node_labels, root_ids, root_labels): #not used
        batch.ndata['h']=self.embeddings(batch.ndata['name'])
        #batch.ndata['h']=self.cell.position_encoding.position_encoding_by_index(batch.ndata['h'],dfs_positions) #dfs position encoding

        for i in range(self.num_stacks):
            dgl.prop_nodes_topo(batch,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            '''if self.top_down:
                prop_nodes_topdown(batch,
                                    message_func=self.cell_topdown.message_func,
                                    reduce_func=self.cell_topdown.reduce_func,
                                    apply_node_func=self.cell_topdown.apply_node_func)'''
            #print(batch.ndata['h'].size(),node_ids.size(),root_ids.size())
        h=batch.ndata.pop('h')
        h=F.dropout(h,p=0.5)
        node_hs=h[node_ids]
        nodes_pred=self.classifier(node_hs)
        #print(nodes_pred.size())
        '''if self.pool_mode=='root':
            batch_pred=batch.ndata['h'][root_ids]
        else:
            batch_pred=self.pooling(batch,batch.ndata['h'])
        #print(batch_pred.size())'''
        sents_hs=h[root_ids]
        #sents_hs=self.pooling(batch,batch.ndata['h'])
        sents_pred=self.classifier(sents_hs)
        return nodes_pred,sents_pred


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
        #print(batch.ndata['type'].size())
        batch.ndata['h']=self.embeddings(batch.ndata['type'])
        #dfs_positions=batch.ndata['dfs_id']
        #batch.ndata['h']=self.cell.position_encoding.position_encoding_by_index(batch.ndata['h'],dfs_positions) #dfs position encoding
        #batch.ndata['q']=self.cell.q_linear(batch.ndata['h'])
        #batch.ndata['q']=batch.ndata['q'].view(-1,self.num_heads,self.cell.d_k)
        dgl.prop_nodes_topo(batch,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        if self.top_down:
            prop_nodes_topdown(batch,
                                message_func=self.cell_topdown.message_func,
                                reduce_func=self.cell_topdown.reduce_func,
                                apply_node_func=self.cell_topdown.apply_node_func)
            #quit()
        return batch

class TreeTransformer_localize(torch.nn.Module):
    def __init__(self, num_heads, dim_model, dim_hidden, vocab_size, dropout=0.2):
        super(TreeTransformer_localize, self).__init__()
        self.encoder=TreeTransformerEncoder(num_heads, dim_model, dim_hidden, vocab_size, dropout)
        self.cls = nn.Linear(dim_model, 1)

    def forward(self, batch,batch_maxlen,batch_numnodes):
        batch=self.encoder(batch)
        batch.ndata['prob']=self.cls(batch.ndata['h']).squeeze(1)
        batch.ndata['prob'] = batch.ndata['prob'].masked_fill(batch.ndata['is_op'] == False, -1e9) #mask non-operator
        
        probs=batch.ndata['prob']
        graph_probs=list(probs.split(batch_numnodes))
        graph_probs=pad_sequence(graph_probs,batch_first=True,padding_value=-1e9)
        graph_probs=F.log_softmax(graph_probs,dim=1)

        '''probs= dgl.softmax_nodes(batch, feat='prob') #apply softmax_nodes without log
        #probs=probs.log()
        graph_probs=list(probs.split(batch_numnodes))
        graph_probs=pad_sequence(graph_probs,batch_first=True,padding_value=0)'''

        '''for i in range(len(graph_probs)):
            graph_probs[i]=torch.cat([graph_probs[i],torch.zeros(batch_maxlen-batch_numnodes[i]).to(batch.device)])
            print(graph_probs[i].size())
        graph_probs=torch.stack(graph_probs)'''
        #print(graph_probs.size())
        return graph_probs

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

    def forward(self, batch,root_ids=None):
        batch.ndata['h']=torch.cat([self.type_embeddings(batch.ndata['type']),self.token_embeddings(batch.ndata['token'])],dim=1)
        #dfs_positions=batch.ndata['dfs_id']
        #batch.ndata['h']=self.cell.position_encoding.position_encoding_by_index(batch.ndata['h'],dfs_positions) #dfs position encoding

        for i in range(self.num_stacks):
            dgl.prop_nodes_topo(batch,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            if self.top_down:
                prop_nodes_topdown(batch,
                                    message_func=self.cell_topdown.message_func,
                                    reduce_func=self.cell_topdown.reduce_func,
                                    apply_node_func=self.cell_topdown.apply_node_func)

        if self.pool_mode=='root':
            batch_pred=batch.ndata['h'][root_ids]
        else:
            batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_pred=self.classifier(batch_pred)
        return batch_pred


class TreeTransformer_codesearch(torch.nn.Module): #not used
    def __init__(self, num_heads, dim_model, d_k, d_v, dim_hidden, vocab_size, dropout=0.2, num_stacks=1):
        super(TreeTransformer_codesearch, self).__init__()
        self.num_heads=num_heads
        self.dim_model=dim_model
        self.dropout = torch.nn.Dropout(dropout)
        self.cell=TreeTransformerCell(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout)
        self.cell_topdown=TreeTransformerCell_topdown(num_heads,dim_model,dim_ff=dim_hidden,dropout=dropout)
        self.embeddings=nn.Embedding(vocab_size,dim_model)
        self.num_stacks = num_stacks
        self.pooling=GlobalAttentionPooling(nn.Linear(dim_model,1))
        self.top_down=True
        print('top_down',self.top_down)
        self.pool_mode='attention'
        print(self.pool_mode)
        self.nlencoder=BasicTransformerEncoder(self.embeddings, dim_model, num_heads, 1024, nlayers=4, dropout=dropout)
        self.dense = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
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

    def outputlinear(self,encodings):
        encodings=encodings[0,:,:] #get output of the first token
        pooled_output = self.dense(encodings)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def forward(self, batch,nlbatch, nl_padding_mask, root_ids=None):
        bs=nlbatch.size()[0]
        batch.ndata['h']=self.embeddings(batch.ndata['type'])
        dfs_positions=batch.ndata['dfs_id']
        #batch.ndata['h']=self.cell.position_encoding.position_encoding_by_index(batch.ndata['h'],dfs_positions) #dfs position encoding

        for i in range(self.num_stacks):
            dgl.prop_nodes_topo(batch,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            if self.top_down:
                prop_nodes_topdown(batch,
                                    message_func=self.cell_topdown.message_func,
                                    reduce_func=self.cell_topdown.reduce_func,
                                    apply_node_func=self.cell_topdown.apply_node_func)

        if self.pool_mode=='root':
            batch_pred=batch.ndata['h'][root_ids]
        else:
            batch_pred=self.pooling(batch,batch.ndata['h'])
        nlbatch=nlbatch.transpose(0,1)
        nl_encodings=self.nlencoder.encoding(nlbatch,src_key_padding_mask=nl_padding_mask)
        scores=(nl_encodings[:,None,:]*batch_pred[None,:,:]).sum(-1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,batch_pred,nl_encodings

class Transformer_codesearch(torch.nn.Module): #not used
    def __init__(self, num_heads, dim_model, d_k, d_v, dim_hidden, vocab_size, dropout=0.2, num_layers=2):
        super(Transformer_codesearch, self).__init__()
        self.num_heads=num_heads
        self.dim_model=dim_model
        self.dropout = torch.nn.Dropout(dropout)
        self.embeddings=nn.Embedding(vocab_size,dim_model)
        self.num_layers=num_layers
        self.pool_mode='attention'
        print(self.pool_mode)
        self.encoder=BasicTransformerEncoder(self.embeddings, dim_model, num_heads, dim_hidden, num_layers, dropout=dropout)
        #self.nlencoder=BasicTransformerEncoder(self.embeddings, dim_model, num_heads, dim_hidden, num_layers, dropout=dropout)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, codebatch,nlbatch,code_padding_mask,nl_padding_mask):
        bs=nlbatch.size()[0]
        codebatch=codebatch.transpose(0,1)
        code_encodings=self.encoder.encoding(codebatch,src_key_padding_mask=code_padding_mask)
        nlbatch=nlbatch.transpose(0,1)
        nl_encodings=self.encoder.encoding(nlbatch,src_key_padding_mask=nl_padding_mask)
        scores=(nl_encodings[:,None,:]*code_encodings[None,:,:]).sum(-1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_encodings,nl_encodings
