import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1,2)) 
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


if __name__ == '__main__':

    k = torch.Tensor([[10.,0,0],
                    [0,10,0],
                    [0,0,10],
                    [0,0,10]]) # 4 x 3
    
    v = torch.Tensor([[1.,0],
                    [10,0],
                    [100,5],
                    [1000,6]]) # 4 x 2
    
    # q = torch.Tensor([[0., 0, 10]]) # 1 x 3
    q = torch.Tensor([[0., 0, 10], 
                    [0., 10, 0], 
                    [10., 10, 0]]) # 3 x 3

    d_k = k.size(-1)

    sda = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
    output, attn = sda(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)) # b x seq x dim      