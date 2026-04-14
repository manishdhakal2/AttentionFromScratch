import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()

        #Assumes dimension of key is equal to dimension of value

        self.embed_dim = embed_dim

        #Dimension for the weights
        self.D_w = 64

        #Initialize the weights
        self.w_Q = nn.Parameter(torch.empty(self.embed_dim, self.D_w))
        self.w_K =  nn.Parameter(torch.empty(self.embed_dim, self.D_w))
        self.w_V = nn.Parameter(torch.empty(self.embed_dim, self.D_w))

        #Apply xavier initialization
        nn.init.xavier_uniform_(self.w_Q)
        nn.init.xavier_uniform_(self.w_K)
        nn.init.xavier_uniform_(self.w_V)
    
    def forward(self, embeddings : torch.Tensor) -> torch.Tensor:

        """
        Returns the context vector generated from the given embeddings 

        Params :
        embedding (B,V,E) : 
        """

        query = embeddings @ self.w_Q
        key = embeddings @ self.w_K
        value = embeddings @ self.w_V
        d_k = key.shape[2]

        #Compute dot product
        dot_qk = query @ key.permute((0,2,1))


        #Normalize the dot product
        normalized_dot_qk = dot_qk/d_k **0.5

        #Softmax across tokens  (dim = -1)
        softmaxed_dot_qk = torch.softmax(normalized_dot_qk, dim = -1)

        #Get the final attention scores
        attention_qkv = softmaxed_dot_qk @ value

        return attention_qkv








        