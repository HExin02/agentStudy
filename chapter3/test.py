from Transformer import MultiHeadAttention
import torch
mulAtt = MultiHeadAttention(768,8)
q = torch.rand((1,512,768))
k = torch.rand((1,512,768))
v = torch.rand((1,512,768))
x = mulAtt(q,k,v)