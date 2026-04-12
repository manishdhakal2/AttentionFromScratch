
import numpy as np

class Attention:
    def __init__(self):

        self.general_w = None
        self.concat_w = None
        self.concat_v = None
        pass

    def dot_attention(self,h_enc,h_dec_t):

        """
        Returns the dot product attention of the given encoder and decoder hidden states

        Params :
            h_enc (B,T,H): Encoder hidden state across all timesteps
            h_dec_t(B,H) :  Current Decoder hidden state  

        """

        B,T,H = h_enc.shape

        similarity_score = np.zeros((B,T))

        for index in range(B):
            similarity_score[index] = np.dot(h_enc[index], h_dec_t[index])


        return self.get_context_vector(h_enc, similarity_score)

    def general_attention(self,h_enc, h_dec_t):
        """
        Returns the general weighted  attention of the given encoder and decoder hidden states

        Params :
            h_enc (B,T,H): Encoder hidden state across all timesteps
            h_dec_t(B,H) :  Current Decoder hidden state  

        """
        
        B,T,H = h_enc.shape

        #Initialize the weight matrix randomly
        if self.general_w is None:
            self.general_w = np.random.uniform(0,1,(H,H)) 

        transformed_h_dec = h_dec_t @ self.general_w
        
        similarity_score = np.zeros((B,T))

        for index in range(B):
            similarity_score[index] = np.dot(h_enc[index],  transformed_h_dec[index])
        
        return self.get_context_vector(h_enc, similarity_score)
    


    def concat_attention(self, h_enc, h_dec_t):
        """
        Returns the general weighted  attention of the given encoder and decoder hidden states

        Params :
            h_enc (B,T,H): Encoder hidden state across all timesteps
            h_dec_t(B,H) :  Current Decoder hidden state  

        """
        D = 64
        B,T,H = h_enc.shape

        if self.concat_w is None:
            self.concat_w = np.random.uniform(0,1,(2*H,D))
        
        #Make (B,H) -> (B,1,H) and repeat T times -> (B,T,H)
        repeated_h_dec = np.repeat(h_dec_t[:,None,:], T, axis =1)

        #Concat Repeated h_dec and h_enc
        joint_states = np.concatenate([repeated_h_dec, h_enc],axis = 2)

        #Multiply W and [Q;K]
        transformed_states = joint_states @ self.concat_w

        if self.concat_v is  None:
            self.concat_v = np.random.uniform(0,1,(D,))

        #Get similarity score
        similarity_score = transformed_states @ self.concat_v

        return self.get_context_vector(h_enc, similarity_score)



    def softmax(self,x):

        """
        Returns the softmax of the given input vector
        """
        return np.exp(x)/np.sum(np.exp(x), axis =1, keepdims = True)


    def get_context_vector(self, h_enc, similarity_score):
        B,T,H = h_enc.shape

        softmaxed_score = self.softmax(similarity_score)

        context_vector = np.zeros((B,H))

        for index in range(B):
            context_vector[index] = np.dot(softmaxed_score[index], h_enc[index])


        return context_vector






