from typing import Optional
import torch
from torch import nn
import torch.utils.data
import math
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Graphomer_20220111(nn.Module):
    def __init__(self):
        super(Graphomer_20220111,self).__init__()

        self.encoder_layer = nn.modules.transformer.TransformerEncoderLayer(d_model=120,nhead=10,dim_feedforward=1024,dropout=0.05,batch_first=True)
        self.encoder_norm = nn.modules.normalization.LayerNorm(normalized_shape=120)
        self.encoder = nn.modules.transformer.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=6,norm=self.encoder_norm)

        self.embedding_src = nn.modules.sparse.Embedding(9,3)
        self.embedding_tgt = nn.modules.sparse.Embedding(22,3)
        self.position_encoder = PositionalEncoding(emb_size=120,dropout=0.05)
        self.encoder_generator = nn.modules.linear.Linear(120,22)

        self.src_mask_encoding1 = nn.modules.linear.Linear(10,20)
        self.src_mask_encoding2 = nn.modules.linear.Linear(20,10)
        self.src_encoding = nn.modules.linear.Linear(12,120)

    def forward(self, phipsi: Tensor, DSSP: Tensor, centerity: Tensor , tgt: Tensor, rand_seed: Tensor, src_mask: Optional[Tensor] = None,padding_mask: Optional[Tensor] = None):
        emd_DSSP = self.embedding_src(DSSP)
        sin_phipsi = torch.sin(phipsi)
        cos_phipsi = torch.cos(phipsi)
        emd_tgt = self.embedding_tgt(tgt)
        emd_src = torch.cat([sin_phipsi,cos_phipsi,emd_DSSP,centerity.unsqueeze(-1),emd_tgt,rand_seed],dim=-1) #2+2+3+1+3+1=12
        emd_src = self.src_encoding(emd_src)
        emd_src = self.position_encoder(emd_src)
        if(src_mask is not None):
            src_mask = self.src_mask_encoding2(self.src_mask_encoding1(src_mask))
            #print(src_mask.shape)
            src_mask = src_mask.transpose(dim0=-1,dim1=1)
            #print(src_mask.shape)
            src_mask = src_mask.reshape(src_mask.shape[0]*src_mask.shape[1],src_mask.shape[2],src_mask.shape[3])
            memory = self.encoder(emd_src, mask=src_mask, src_key_padding_mask=padding_mask)
        else:
            memory = self.encoder(emd_src,src_key_padding_mask=padding_mask)
        encoder_output = self.encoder_generator(memory)
        return encoder_output
            

#compute the design accuracy
def accuracy(predict,label):
    count = 0
    count_0 = 0
    for i in range(0,len(predict[1])):
        if(label[i] == 0):
            count_0 = count_0 + 1
        elif (predict[1][i] == label[i]):
            count = count+1
    return count/(len(predict[1]) - count_0)


