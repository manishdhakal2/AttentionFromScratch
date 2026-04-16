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
        super().__init__()
        self.embed_dim = embed_dim
        self.h = head_size

        #Initialize the dim for each head
        self.h_d = int(model_dim/head_size)

        #Initialize the matrix for Q K and V
        self.w_Q = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))
        self.w_K =  nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))
        self.w_V = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))

        #Apply xavier initialization
        nn.init.xavier_uniform_(self.w_Q)
        nn.init.xavier_uniform_(self.w_K)
        nn.init.xavier_uniform_(self.w_V)

        

    def forward(self, x):

        B, T, E = x.shape

       
        if not E == self.embed_dim:
            raise RuntimeError("Embedding size must be equal to model size")

        #Compute Q,K,V. Shape : (B,T,E)
        query = x @ self.w_Q
        key = x @ self.w_K
        value = x @ self.w_V


        #Return "h" split views of query and transpose in order to get an independent matrix for each head
        #Shape : (B,h,T,h_d)
        split_query = query.view((B,T,self.h,self.h_d)).transpose(1,2)
        split_key = key.view((B,T,self.h,self.h_d)).transpose(1,2)
        split_value = value.view((B,T,self.h,self.h_d)).transpose(1,2)

        #Compute and normalize the similarity score.  Shape :(B,h,T,T)
        dot_qk = split_query @ split_key.transpose(2,3)
        normalized_dot_qk = dot_qk/self.h_d **0.5
        softmaxed_dot_qk = torch.softmax(normalized_dot_qk, dim =-1)

        #Get the attention scores per each head. SHape :(B,h,T,h_d)
        attention_qkv = softmaxed_dot_qk @ split_value

        #Convert the attention score for each head back to each token. Shape:(B,T,h,h_d)
        attention_per_token = torch.einsum("BHTD->BTHD", attention_qkv)

        #Combine each head of token. Shape : (B,H,E)
        combined_attention = attention_per_token.contiguous().view((B,T,self.h*self.h_d))

        return combined_attention


            
        
        
        
