import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_k) -> None:
        super().__init__()

        #Assumes dimension of key is equal to dimension of value
        self.d_k = None

        #Dimension for the weights
        self.D_w = 64

        #Initialize the weights
        self.w_Q, self.w_K, self.w_V = None,None,None
    
    def forward(self, embeddings : torch.Tensor) -> torch.Tensor:

        """
        Returns the context vector generated from the given embeddings 

        Params :
        embedding (B,V,E) : 
        """

        B, V, E = embeddings.shape

        if self.w_Q is None:
            self.w_Q = torch.rand([E, self.D_w], requires_grad=True) 
        if self.w_K is None:
            self.w_K = torch.rand([E, self.D_w], requires_grad= True)
        if self.w_V is None:
            self.w_V = torch.rand([E, self.D_w], requires_grad= True)

        query = embeddings @ self.w_Q
        key = embeddings @ self.w_K
        value = embeddings @ self.w_V
        self.d_k = torch.tensor(key.shape[2])

        #Compute dot product
        dot_qk = query @ key.permute((0,2,1))


        #Normalize the dot product
        normalized_dot_qk = dot_qk/torch.sqrt(self.d_k)

        #Softmax across tokens  (dim = 1)
        softmaxed_dot_qk = torch.softmax(normalized_dot_qk, dim = 1)

        #Get the final attention scores
        attention_qkv = softmaxed_dot_qk @ value.permute((0,2,1))

        return attention_qkv








        