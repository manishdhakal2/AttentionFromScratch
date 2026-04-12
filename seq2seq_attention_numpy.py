
import numpy as np

class Attention:
    def __init__(self):


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


        softmaxed_score = self.softmax(similarity_score)

        context_vector = np.zeros((B,H))

        for index in range(B):
            context_vector[index] = np.dot(softmaxed_score[index], h_enc[index])


        return context_vector



    def softmax(self,x):

        """
        Returns the softmax of the given input vector
        """
        return np.exp(x)/np.sum(np.exp(x), axis =1, keepdims = True)


