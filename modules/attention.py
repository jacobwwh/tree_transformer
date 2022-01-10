import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention_TUPE(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, top_down=False, tupe=False, pe_mode='learned'):
        super(MultiHeadAttention_TUPE, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len=10000

        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.top_down=top_down
        self.TUPE=tupe
        if self.TUPE==True:
            self.U_q = nn.Linear(d_model, d_model, bias=False)
            self.U_k = nn.Linear(d_model, d_model, bias=False)
            if pe_mode=='fixed':
                self.pe = torch.zeros(self.max_len, d_model)
                position = torch.arange(0, self.max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) *
                                    -(math.log(10000.0) / d_model))
                self.pe[:, 0::2] = torch.sin(position * div_term)
                self.pe[:, 1::2] = torch.cos(position * div_term) #size: (max_len, d_model)
                self.pe.requires_grad=False
            else:
                self.pe=nn.Parameter(torch.rand([self.max_len,d_model])) #learned position encoding
            self.pe.data.uniform_(-0.1, 0.1)
            self.scale_factor = np.sqrt(2*d_k) #same as the TUPE paper

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)
        len_q=q.size(1)
        len_k=k.size(1)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = q.view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = k.view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = v.view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) 
        if self.TUPE==True:
            q_pe=self.pe[:len_q].to(q.device)
            k_pe=self.pe[:len_k].to(q.device)
            q_pe=self.U_q(q_pe)
            k_pe=self.U_k(k_pe)
            pe_attention=torch.matmul(q_pe, k_pe.transpose(-1, -2))
            scores=scores+pe_attention.unsqueeze(0)
        scores=scores/self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)

        attn = self.dropout(self.softmax(scores))
        context = torch.matmul(attn, v)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn


class MultiHeadAttention_TUPE_parent(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, top_down=False, tupe=False, pe_mode='learned'):
        super(MultiHeadAttention_TUPE_parent, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len=10000

        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.top_down=top_down
        self.TUPE=tupe
        if self.TUPE==True:
            self.U_q = nn.Linear(d_model, d_model, bias=False)
            self.U_k = nn.Linear(d_model, d_model, bias=False)
            if pe_mode=='fixed':
                self.pe = torch.zeros(self.max_len, d_model)
                position = torch.arange(0, self.max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) *
                                    -(math.log(10000.0) / d_model))
                self.pe[:, 0::2] = torch.sin(position * div_term)
                self.pe[:, 1::2] = torch.cos(position * div_term) #size: (max_len, d_model)
                self.pe.requires_grad=False
            else:
                self.pe=nn.Parameter(torch.rand([self.max_len,d_model])) #learned position encoding
            self.pe.data.uniform_(-0.1, 0.1)
            self.scale_factor = np.sqrt(2*d_k) #same as the TUPE paper

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)
        len_q=q.size(1) #len_q=1
        len_k=k.size(1)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = q.view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = k.view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = v.view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) 
        if self.TUPE==True:
            q_pe=torch.ones(1,self.d_model,device=q.device)
            k_pe=self.pe[:len_k]
            q_pe=self.U_q(q_pe)
            k_pe=self.U_k(k_pe)
            pe_attention=torch.matmul(q_pe, k_pe.transpose(-1, -2))
            scores=scores+pe_attention.unsqueeze(0)
        scores=scores/self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)

        attn = self.dropout(self.softmax(scores))
        context = torch.matmul(attn, v)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn
