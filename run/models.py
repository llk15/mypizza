import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

#################### Linear ####################
class MyModelA(nn.Module): # one layer, A+B, commutative
    def __init__(self, cfg):
        super(MyModelA, self).__init__()
        n_vocab = cfg.n_vocab
        d_hidden = cfg.d_model
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.embed.weight.data /= (d_hidden//2)**0.5
        if cfg.tied_embeddings:
            self.unembed = self.embed
        else:
            self.unembed = nn.Embedding(n_vocab, d_hidden)
            self.unembed.weight.data /= d_hidden**0.5
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0]+x[:,1])
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
    
class MyModelB(nn.Module): # two layers, A+B, commutative
    def __init__(self, cfg):
        super(MyModelB, self).__init__()
        n_vocab = cfg.n_vocab
        d_hidden = cfg.d_model
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.embed.weight.data /= (d_hidden//2)**0.5
        if cfg.tied_embeddings:
            self.unembed = self.embed
        else:
            self.unembed = nn.Embedding(n_vocab, d_hidden)
            self.unembed.weight.data /= d_hidden**0.5
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.l2 = nn.Linear(d_hidden, d_hidden, bias=True)
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0]+x[:,1])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
    
class MyModelC(nn.Module): # two layers, l1(A) + l1(B), commutative
    def __init__(self, cfg):
        super(MyModelC, self).__init__()
        n_vocab = cfg.n_vocab
        d_hidden = cfg.d_model
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.embed.weight.data /= (d_hidden//2)**0.5
        if cfg.tied_embeddings:
            self.unembed = self.embed
        else:
            self.unembed = nn.Embedding(n_vocab, d_hidden)
            self.unembed.weight.data /= d_hidden**0.5
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.l2 = nn.Linear(d_hidden, d_hidden, bias=True)
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0])+self.l1(x[:,1])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
    
class MyModelD(nn.Module): # one layer, [A,B], noncommutative
    def __init__(self, cfg):
        super(MyModelD, self).__init__()
        n_vocab = cfg.n_vocab
        d_hidden = cfg.d_model
        self.embed = nn.Embedding(n_vocab, d_hidden//2)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = torch.cat([x[:,0],x[:,1]], dim=1)
        x = self.l1(x)
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x

class MyModelX(nn.Module):  ## two embeddings, noncommutative
    def __init__(self, cfg):
        n_vocab = cfg.n_vocab
        d_hidden = cfg.d_model
        super(MyModelX, self).__init__()
        self.embed1 = nn.Embedding(n_vocab, d_hidden)
        self.embed2 = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed1.weight.data /= (d_hidden//2)**0.5
        self.embed2.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.l1(self.embed1(x[:,0])+self.embed2(x[:,1]))
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x


#################### transformers ####################

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, attn_coeff):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.attn_coeff = attn_coeff
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked =attn_scores_pre
        normalized = attn_scores_masked/np.sqrt(self.d_head)
        normalized = F.softmax(normalized, dim=-1)
        attn_matrix = normalized*self.attn_coeff+(1-self.attn_coeff)
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        assert act_type in ['relu', 'gelu', 'tanh']
        
    def forward(self, x):
        x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        if self.act_type=='relu':
            x = F.relu(x)
        elif self.act_type=='gelu':
            x = F.gelu(x)
        elif self.act_type=='tanh':
            x = F.tanh(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

class MyLinear(nn.Module):
    def __init__(self, d_model, act_type):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_model, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        assert act_type in ['relu', 'gelu', 'tanh']
        
    def forward(self, x):
        x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        if self.act_type=='relu':
            x = F.relu(x)
        elif self.act_type=='gelu':
            x = F.gelu(x)
        elif self.act_type=='tanh':
            x = F.tanh(x)
        return x
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, num_heads, n_ctx, act_type, attn_coeff=1.0):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, attn_coeff=attn_coeff)
        self.mlp = MLP(d_model, d_model*4,act_type)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_ctx = 2

        n_vocab = cfg.n_vocab
        num_heads = cfg.n_heads
        d_model = cfg.d_model
        d_head = cfg.d_model//cfg.n_heads

        self.embed = nn.Embedding(n_vocab, d_model)
        self.embed.weight.data /= d_model**0.5
        self.pos_embed = PosEmbed(n_ctx, d_model)
        if cfg.tied_embeddings:
            self.unembed = self.embed
        else:
            self.unembed = nn.Embedding(n_vocab, d_model)
            self.unembed.weight.data /= d_model**0.5

        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_head, num_heads, n_ctx, cfg.act_fn) for i in range(cfg.n_layers)])
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        assert len(x.shape)==3 #[batch, token, n_model] 
        for blk in self.blocks:
            x = blk(x)
        x = x @ self.unembed.weight.t()
        return x
    


    #     self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
    # def backdoor(self, x):
        # x = self.embed(x)
        # assert len(x.shape)==3 and x.shape[1]==2
    #     x = self.l1(x[:,0]+x[:,1])
    #     x = F.relu(x)
    #     return x
    # def forward(self, x):
    #     x = self.backdoor(x)
    #     x = x @ self.unembed.weight.t()
    #     return x
    
    

