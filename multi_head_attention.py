import torch 
import torch.nn as nn

from attention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, embed_dim:int, head_size :int) -> None:
        """
        Params :
        head_size (int) : The number of heads in the attention block
        embed_dim (int) : The size of each embedding
        """

        self.embed_dim = embed_dim
        self.head_size = head_size

        #Initialize the model dim for each head
        self.D_wi = int(model_dim/head_size)

        #Create a list of independednt attention heads
        self.heads = nn.ModuleList([SelfAttention(self.D_wi, self.embed_dim) for i in range(head_size)] )

        

    def forward(self, x):

        pass
        
